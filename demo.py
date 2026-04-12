import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as tf
from matplotlib import colormaps

from opensplat3d.data.colmap_loader import qvec2rotmat
from opensplat3d.gaussian_model import GaussianModel, sub_gaussians
from opensplat3d.gaussian_renderer import render
from opensplat3d.language import LanguageModel
from opensplat3d.params import PipeParams
from opensplat3d.scene.camera import Camera, ViewerCamera
from opensplat3d.utils.camera_utils import (
    focal2fov,
    fov2focal,
    get_projection_matrix,
    get_world2view,
)
from opensplat3d.utils.setup_utils import setup
from opensplat3d.utils.sh_utils import SH2RGB
from opensplat3d.utils.vis_utils import enhance_image, pca


def get_c2w(camera: viser.CameraHandle):
    c2w = np.eye(4, dtype=np.float32)
    wxyz: torch.FloatTensor = torch.from_numpy(camera.wxyz)  # type: ignore
    c2w[:3, :3] = qvec2rotmat(wxyz).numpy()
    c2w[:3, 3] = camera.position
    return c2w


def get_w2c(camera: viser.CameraHandle):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c


VISER_SCALE: float = 1.0
LANGUAGE_INFO_TEMPLATE = """
**Similarity**
- min: {min:.4f}
- max: {max:.4f}
- avg: {avg:.4f}

**Labels**
- {labels}
"""
LANGUAGE_COLOR_MAP = "viridis"


@dataclass
class ClickSelection:
    gaussians: GaussianModel
    mask: torch.Tensor
    diff: torch.Tensor  # diff between feature and Gaussians
    feature: torch.Tensor  # clicked feature
    xy: tuple[float, float]  # (x,y) coordinate of selection


@dataclass
class TextPromptSelection:
    gaussians: GaussianModel
    mask: torch.Tensor
    sim: torch.Tensor  # similarity
    prompt: str


@dataclass
class LabelSelection:
    gaussians: GaussianModel
    mask: torch.Tensor
    label: int


class ViserViewer:
    def __init__(
        self,
        model_dir: str,
        vis_cameras: bool = True,
        language: str | None = None,
        initial_resolution: int = 1240,
    ) -> None:
        self.model_dir = Path(model_dir)
        setup_params = setup(self.model_dir, load_masks=False)

        self.setup_params = setup_params
        self.need_update = False

        self.gaussians = setup_params.gaussians
        self.pipe = PipeParams()
        self.bg_colors = {
            "black": torch.tensor([0, 0, 0], dtype=torch.float32).cuda(),
            "white": torch.tensor([1, 1, 1], dtype=torch.float32).cuda(),
        }
        self.vis_cameras = vis_cameras
        self.cameras = setup_params.scene.get_train_cameras()

        device = self.gaussians._xyz.device
        model_path = setup_params.model_path
        label_path = model_path / "clustering" / "labels.npy"
        if label_path.exists():
            self.labels = torch.from_numpy(np.load(label_path)).to(device)
        else:
            self.labels = None

        self.lang_model: LanguageModel | None = None
        self.inst_lang_embeds: torch.Tensor | None = None

        if language is not None:
            print("Loading language model...")
            self.lang_model = LanguageModel(language)
            lang_embedding_path = model_path / f"{language}_embeddings.pth"
            assert lang_embedding_path.exists(), (
                f"Embeddings not found at {lang_embedding_path}"
            )
            assert self.labels is not None
            embeds_data: dict = torch.load(
                lang_embedding_path,
                weights_only=False,
            )
            embeds_data.pop("config", None)

            valid_clusters = embeds_data["valid"]
            self.inst_lang_embeds = torch.from_numpy(
                embeds_data["embeddings"][valid_clusters]
            ).cuda()
            unique_labels = torch.unique(self.labels)
            unique_labels = unique_labels[unique_labels != -1]
            assert unique_labels.shape[0] == valid_clusters.shape[0]
            for label, valid in zip(unique_labels, valid_clusters):
                if not valid:
                    self.labels[self.labels == label] = -1
            print(f"Remove invalid labels: {(~valid_clusters).sum()}")

        self.cluster_colors: torch.Tensor | None = None
        self.pca_colors: torch.Tensor | None = None
        self.sorted_labels: torch.Tensor | None = None
        if self.gaussians.get_features is not None and self.labels is not None:
            features = self.gaussians.get_features.detach()[
                ..., : setup_params.model_params.mask_dim
            ]
            rng = np.random.default_rng(42)

            labels_, counts = torch.unique(self.labels, return_counts=True)
            sort_index = torch.argsort(counts, descending=True)
            labels_ = labels_[sort_index]
            self.sorted_labels = labels_
            random_colors = (
                torch.from_numpy(rng.random((self.labels.shape[0], 3)))
                .float()
                .to(device)
            )
            cluster_color = random_colors[self.labels]
            cluster_color[self.labels == -1] = 0
            self.cluster_colors = cluster_color

            pca_colors = pca(features.cpu().numpy(), 3, normalize=True)
            pca_colors = (pca_colors * 255).astype(np.uint8)
            pca_colors = enhance_image(pca_colors[None, ...])[0]
            self.pca_colors = torch.from_numpy(pca_colors).float().div(255.0).to(device)

        self.selection: ClickSelection | TextPromptSelection | LabelSelection | None = (
            None
        )
        self.active_animation = False

        self.frustum_handles: list[viser.CameraFrustumHandle] = []

        self.server = viser.ViserServer()
        self.server.gui.configure_theme(dark_mode=True)

        self.server.scene.set_up_direction("-y")

        # Render section
        self.resolution_slider = self.server.gui.add_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=initial_resolution
        )

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        # self.original_aspect_ratio_checkbox = self.server.gui.add_checkbox(
        #     "Original Aspect Ratio",
        #     initial_value=False,
        # )

        # @self.original_aspect_ratio_checkbox.on_update
        # def _(_):
        #     self.need_update = True

        self.scale_slider = self.server.gui.add_slider(
            "Scale", 0, 1, step=0.01, initial_value=1.0
        )

        @self.scale_slider.on_update
        def _(_: viser.GuiEvent[viser.GuiSliderHandle[float]]):
            self.need_update = True

        self.bg_color_checkbox = self.server.gui.add_checkbox(
            "White Background", initial_value=False
        )

        @self.bg_color_checkbox.on_update
        def _(_):
            self.need_update = True

        self.render_isolated_checkbox = self.server.gui.add_checkbox(
            "Isolated",
            initial_value=False,
            hint="Render Gaussians isolated by removing all other Gaussians that do not belong to the selection.",
        )

        @self.render_isolated_checkbox.on_update
        def _(_: viser.GuiEvent[viser.GuiCheckboxHandle]):
            self.need_update = True

        if self.labels is not None and self.sorted_labels is not None:
            vlm_descriptions = {}
            if language is not None:
                desc_path = model_path / f"{language}_descriptions.pth"
                if desc_path.exists():
                    try:
                        desc_data = torch.load(desc_path, weights_only=False)
                        for k, v in desc_data.items():
                            vlm_descriptions[k] = v["descriptions"][0]
                    except Exception as e:
                        print(f"Could not load vlm semantic descriptions from {desc_path}: {e}")

            options = ["all"]
            if -1 in self.sorted_labels.tolist():
                options.append("!= -1")
            
            self.label_mapping = {}
            for lbl in self.sorted_labels.tolist():
                if lbl in vlm_descriptions:
                    desc_text = vlm_descriptions[lbl]
                    if len(desc_text) > 60:
                        desc_text = desc_text[:57] + "..."
                    opt_str = f"#{lbl}: {desc_text}"
                else:
                    opt_str = str(lbl)
                options.append(opt_str)
                self.label_mapping[opt_str] = lbl

            self.label_dropdown = self.server.gui.add_dropdown(
                "Label",
                options=options,
                initial_value="all",
                hint="Render cluster labels",
            )

            @self.label_dropdown.on_update
            def _(event: viser.GuiEvent[viser.GuiDropdownHandle[str]]):
                if self.labels is None:
                    return
                if event.client is not None:
                    # when setting value on server, the client is none and we want only to trigger this if the client actually selected something
                    with event.client.atomic():
                        self.click_selection_markdown.visible = False
                        if self.query_text_markdown is not None:
                            self.query_text_markdown.visible = False
                    if event.target.value == "all":
                        self.selection = None
                        self.need_update = True
                        return
                    if event.target.value == "!= -1":
                        mask: torch.BoolTensor = self.labels != -1  # type: ignore
                        self.selection = LabelSelection(
                            sub_gaussians(self.gaussians, mask), mask, -2
                        )
                        self.need_update = True
                        return

                    label_val = event.target.value
                    if label_val in self.label_mapping:
                        label = int(self.label_mapping[label_val])
                    else:
                        label = int(label_val)
                        
                    mask: torch.BoolTensor = self.labels == label  # type: ignore
                    self.selection = LabelSelection(
                        sub_gaussians(self.gaussians, mask), mask, label
                    )
                    self.need_update = True

            self.render_dropdown = self.server.gui.add_dropdown(
                "Render",
                options=["RGB", "PCA", "Labels", "Alpha"],
                initial_value="RGB",
            )
        else:
            self.render_dropdown = self.server.gui.add_dropdown(
                "Render",
                options=["RGB", "PCA", "Alpha"],
                initial_value="RGB",
            )

        @self.render_dropdown.on_update
        def _(_: viser.GuiEvent[viser.GuiDropdownHandle[str]]):
            self.need_update = True

        # Click section
        self.tab_group = self.server.gui.add_tab_group()
        if self.labels is not None:
            with self.tab_group.add_tab("Click", icon=viser.Icon.CLICK):
                self.click_selection_button = self.server.gui.add_button(
                    "Click Selection", icon=viser.Icon.CLICK
                )
                self.clear_selection_button = self.server.gui.add_button(
                    "Clear Selection", icon=viser.Icon.CLEAR_ALL
                )
                self.threshold_slider = self.server.gui.add_slider(
                    "Threshold", min=0, max=2, step=0.01, initial_value=0.4
                )
                self.threshold_reset_button = self.server.gui.add_button(
                    "Reset Threshold"
                )
                self.click_selection_markdown = self.server.gui.add_markdown(
                    "", visible=False
                )

            @self.threshold_slider.on_update
            def _(event: viser.GuiEvent[viser.GuiSliderHandle[float]]):
                if self.selection is not None and isinstance(
                    self.selection, ClickSelection
                ):
                    self.update_click_selection(event.target.value, None)

            @self.threshold_reset_button.on_click
            def _(_):
                self.threshold_slider.value = 0.4

            @self.click_selection_button.on_click
            def _(event: viser.GuiEvent[viser.GuiButtonHandle]):
                self.click_selection_button.disabled = True
                # self.selection_info = None
                self.selection = None
                self.need_update = True

                assert event.client is not None

                @event.client.scene.on_pointer_event(event_type="click")
                def _(event: viser.ScenePointerEvent) -> None:
                    self.handle_click(event)
                    event.client.scene.remove_pointer_callback()

                @event.client.scene.on_pointer_callback_removed
                def _():
                    self.click_selection_button.disabled = False

            @self.clear_selection_button.on_click
            def _(_):
                self.selection = None
                self.need_update = True

        # Language section
        self.similarity_map_checkbox: viser.GuiInputHandle[bool] | None = None
        self.activation_fn_checkbox: viser.GuiInputHandle[bool] | None = None
        self.query_text_markdown: viser.GuiMarkdownHandle | None = None
        self.language_field_checkbox: viser.GuiInputHandle[bool] | None = None
        self.clear_prompt_button: viser.GuiButtonHandle | None = None
        if language:
            with self.tab_group.add_tab("Language"):
                self.prompt_textbox = self.server.gui.add_text(
                    "Text Prompt", initial_value=""
                )
                self.query_text_button = self.server.gui.add_button("Query")
                self.clear_prompt_button = self.server.gui.add_button(
                    "Clear Prompt", icon=viser.Icon.CLEAR_ALL
                )
                self.similarity_threshold = self.server.gui.add_slider(
                    "Threshold", min=0, max=1, step=0.01, initial_value=0.85
                )
                self.activation_fn_checkbox = self.server.gui.add_checkbox(
                    "Activation Function",
                    initial_value=False,
                )

                self.similarity_map_checkbox = self.server.gui.add_checkbox(
                    "Similarity Map",
                    initial_value=True,
                )
                self.query_text_markdown = self.server.gui.add_markdown(
                    "", visible=False
                )

            @self.query_text_button.on_click
            def _(_: viser.GuiEvent[viser.GuiButtonHandle]):
                text = self.prompt_textbox.value.strip()
                if len(text) == 0:
                    self.selection = None
                    if self.query_text_markdown is not None:
                        self.query_text_markdown.visible = False
                    return
                self.text_query(self.prompt_textbox.value)

            @self.clear_prompt_button.on_click
            def _(_):
                self.selection = None
                self.prompt_textbox.value = ""
                self.need_update = True

            @self.activation_fn_checkbox.on_update
            def _(_):
                if self.selection is None or not isinstance(
                    self.selection, TextPromptSelection
                ):
                    return
                self.text_query(self.prompt_textbox.value)

            @self.similarity_map_checkbox.on_update
            def _(_: viser.GuiEvent[viser.GuiCheckboxHandle]):
                self.need_update = True

        # Camera section
        with self.tab_group.add_tab("Camera", icon=viser.Icon.CAMERA):
            if self.vis_cameras:
                self.toggle_camera_button = self.server.gui.add_button(
                    "Toggle Cameras", icon=viser.Icon.CAMERA
                )

                self.animate_camera_path_button = self.server.gui.add_button(
                    "Animate Camera Path", icon=viser.Icon.VECTOR_SPLINE
                )

                @self.toggle_camera_button.on_click
                def _(_):
                    for cam in self.frustum_handles:
                        cam.visible = not cam.visible

                @self.animate_camera_path_button.on_click
                def _(event: viser.GuiEvent[viser.GuiButtonHandle]):
                    assert event.client is not None
                    self.animate_camera_path(event.client)

            self.reset_camera_button = self.server.gui.add_button(
                "Reset", icon=viser.Icon.RESTORE
            )

            @self.reset_camera_button.on_click
            def _(event: viser.GuiEvent[viser.GuiButtonHandle]):
                assert event.client is not None
                if self.active_animation:
                    self.active_animation = False
                    time.sleep(0.1)  # Hack
                self.animate_to_camera(event.client, self.cameras[0])

            self.print_client_camera_info_button = self.server.gui.add_button(
                "Print Client Camera Info", icon=viser.Icon.INFO_CIRCLE
            )

            @self.print_client_camera_info_button.on_click
            def _(event: viser.GuiEvent[viser.GuiButtonHandle]):
                assert event.client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(event.client.camera.wxyz),
                    event.client.camera.position / VISER_SCALE,
                ).inverse()
                print("Translation:", T_world_current.translation().tolist())
                print(
                    "Rotation:",
                    T_world_current.rotation().as_matrix().transpose(0, 1).tolist(),
                )

        if self.vis_cameras:
            self.add_cameras_to_scene()

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            try:
                c2w = tf.SE3.from_rotation_and_translation(
                    tf.SO3.from_matrix(self.cameras[0].R.transpose(0, 1).numpy()),
                    self.cameras[0].T.numpy(),
                ).inverse()
                with client.atomic():
                    # look_at = c2w @ tf.SE3.from_translation(np.array([0.0, 0.0, -1.0]))
                    # client.camera.fov = self.cameras[0].fovX
                    # client.camera.aspect = (
                    #     self.cameras[0].image_width / self.cameras[0].image_height
                    # )
                    client.camera.up_direction = np.array([0, -1, 0])
                    client.camera.look_at = c2w.translation() * VISER_SCALE
                    # client.camera.look_at = (
                    #     look_at.translation() * VISER_SCALE
                    # )  # c2w.translation() * VISER_SCALE
                    client.camera.wxyz = c2w.rotation().wxyz
                    client.camera.position = c2w.translation() * VISER_SCALE
            except Exception as e:
                print("Client connect:", e)
                traceback.print_stack()
                raise e

            time.sleep(1)
            self.need_update = True

            @client.camera.on_update
            def _(_):
                self.need_update = True

    def update_click_selection(
        self, threshold: float, selection: ClickSelection | None
    ):
        base_selection = selection
        if base_selection is None:
            base_selection = self.selection
            assert base_selection is not None and isinstance(
                base_selection, ClickSelection
            )
            mask: torch.BoolTensor = base_selection.diff < threshold  # type: ignore
            self.selection = ClickSelection(
                sub_gaussians(self.gaussians, mask),
                mask,
                base_selection.diff,
                base_selection.feature,
                base_selection.xy,
            )
        else:
            self.selection = base_selection
            mask = base_selection.mask  # type: ignore

        self.click_selection_markdown.visible = True
        if self.labels is not None:
            labels = self.labels[mask]
            self.click_selection_markdown.content = f"{labels.unique().cpu().numpy()}"
        self.need_update = True

    @torch.no_grad()
    def handle_click(self, event: viser.ScenePointerEvent):
        render_package = self.render(
            event.client, self.resolution_slider.value, self.gaussians
        )
        if (
            render_package is None
            or render_package.features is None
            or self.gaussians.get_features is None
        ):
            return

        x = int(event.screen_pos[0][0] * render_package.features.size(2))
        y = int(event.screen_pos[0][1] * render_package.features.size(1))
        selected_feature = render_package.features[:, y, x]
        diff = self.gaussians.get_features.detach() - selected_feature.unsqueeze(0)
        diff = diff[..., : self.setup_params.model_params.mask_dim]
        diff = diff.norm(2, dim=-1).contiguous()
        mask: torch.BoolTensor = diff < self.threshold_slider.value  # type: ignore
        selection = ClickSelection(
            sub_gaussians(self.gaussians, mask),
            mask,
            diff,
            selected_feature,
            event.screen_pos[0],
        )
        self.update_click_selection(
            self.threshold_slider.value,
            selection,
        )
        self.clear_selection_button.visible = True

    def animate_to_camera(
        self, client: viser.ClientHandle, camera: Camera | viser.CameraFrustumHandle
    ):
        if self.active_animation:
            return
        T_world_current = tf.SE3.from_rotation_and_translation(
            tf.SO3(client.camera.wxyz), client.camera.position / VISER_SCALE
        )
        if isinstance(camera, Camera):
            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3.from_matrix(camera.R.transpose(0, 1).numpy()),
                camera.T.numpy(),
            ).inverse()
        else:
            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(camera.wxyz), camera.position / VISER_SCALE
            )
        T_current_target = T_world_current.inverse() @ T_world_target
        self.active_animation = True
        if not np.all(
            np.isclose(T_world_current.as_matrix(), T_world_target.as_matrix())
        ):
            max_iter = 60
            for j in range(max_iter):
                if not self.active_animation:
                    break
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / (max_iter - 1)
                )
                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation() * VISER_SCALE
                client.flush()  # Optional!
                time.sleep(1 / 240.0)
        self.active_animation = False

    def animate_camera_path(self, client: viser.ClientHandle):
        if self.active_animation:
            return
        self.animate_camera_path_button.disabled = True
        self.animate_to_camera(client, self.cameras[0])

        T_world_current = tf.SE3.from_rotation_and_translation(
            tf.SO3(client.camera.wxyz), client.camera.position / VISER_SCALE
        )
        max_iter = 60
        self.active_animation = True
        for camera in self.cameras[1:]:
            if not self.active_animation:
                break
            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3.from_matrix(camera.R.transpose(0, 1).numpy()),
                camera.T.numpy(),
            ).inverse()
            T_current_target = T_world_current.inverse() @ T_world_target
            for j in range(max_iter):
                if not self.active_animation:
                    break
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / (max_iter - 1)
                )
                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation() * VISER_SCALE
                client.flush()  # Optional!
                time.sleep(1 / (480.0 * 4))
            T_world_current = T_world_set
        self.active_animation = False
        self.animate_camera_path_button.disabled = False

    def add_cameras_to_scene(self):
        def on_camera_click(
            event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle],
        ):
            if self.active_animation:
                self.active_animation = False
                time.sleep(0.1)  # Hack
            self.animate_to_camera(event.client, event.target)

        for camera in self.cameras:
            name = camera.name

            cx = camera.image_width // 2
            cy = camera.image_height // 2

            c2w = tf.SE3.from_rotation_and_translation(
                tf.SO3.from_matrix(camera.R.transpose(0, 1).numpy()),
                camera.T.numpy(),
            ).inverse()

            image = (
                camera.original_image.permute(1, 2, 0).mul(255).to(torch.uint8).numpy()
            )
            frustum_handle = self.server.scene.add_camera_frustum(
                name=f"cameras/{name}",
                fov=camera.fovY,
                scale=0.05 * VISER_SCALE,
                aspect=float(cx / cy),
                wxyz=c2w.rotation().wxyz,
                position=c2w.translation() * VISER_SCALE,
                color=(255, 255, 0),
                image=image,
                jpeg_quality=95,
                visible=False,
            )
            frustum_handle.on_click(on_camera_click)
            self.frustum_handles.append(frustum_handle)

    @torch.no_grad()
    def text_query(self, text: str):
        if self.lang_model is None:
            print("No language model")
            return

        if self.inst_lang_embeds is not None:
            lang_feats = self.inst_lang_embeds
        else:
            print("No language features")
            return

        device = self.gaussians._xyz.device
        text_embedding = self.lang_model.embed_text(
            [self.lang_model.prompt_template.format(text)], True
        ).to(dtype=lang_feats.dtype)

        sim = lang_feats @ text_embedding.T
        sim = self.lang_model.rescale(sim)
        if (
            self.activation_fn_checkbox is not None
            and self.activation_fn_checkbox.value
        ):
            sim = self.lang_model.activation(sim.t()).t()
        sim = sim[:, 0].float()
        mask: torch.BoolTensor = (sim - sim.min()) >= (
            (sim - sim.min()).max() * self.similarity_threshold.value
        )  # type: ignore
        mask = torch.cat([torch.tensor([False], device=mask.device), mask])  # type: ignore
        log_labels = None
        sim_ = None
        if self.inst_lang_embeds is not None:
            # mask needs to be mapped to ids
            assert self.labels is not None
            m = torch.zeros(
                (self.gaussians.num_points,), dtype=torch.bool, device=device
            )
            s = (
                torch.ones(
                    (self.gaussians.num_points,), dtype=torch.float32, device=device
                )
                * sim.min()
            )
            labels = self.labels.unique()
            labels = labels[mask]

            mask = mask[1:]  # type: ignore
            sim_ = sim[mask]

            log_labels = labels
            for label, s2 in zip(labels, sim_):
                m |= self.labels == label
                cscore = s[self.labels == label]
                x = torch.stack([cscore, s2.repeat(cscore.shape[0])], dim=-1)
                s[self.labels == label] = x.max(dim=-1).values
            mask = m  # type: ignore
            sim = s

        if self.query_text_markdown is not None:
            _log_labels = None
            if log_labels is not None:
                if sim_ is None:
                    _log_labels = log_labels.cpu().numpy()
                else:
                    _log_labels = [
                        f"{x}: {y:.2f}" for x, y in zip(labels, sim_.cpu().numpy())
                    ]
            self.query_text_markdown.content = LANGUAGE_INFO_TEMPLATE.format(
                min=sim.min(),
                max=sim.max(),
                avg=sim.mean(),
                labels=_log_labels if _log_labels is not None else "-",
            )
            self.query_text_markdown.visible = True

        if sim.max().item() == 0:
            return
        sim -= sim.min()
        sim /= sim.max()
        color = colormaps.get_cmap(LANGUAGE_COLOR_MAP)(sim.cpu().numpy())[..., :3]
        sim = torch.from_numpy(color).float().to(device)
        self.selection = TextPromptSelection(
            sub_gaussians(self.gaussians, mask), mask, sim, text
        )
        self.need_update = True

    @torch.no_grad()
    def render(
        self,
        client: viser.ClientHandle,
        resolution: int,
        gaussians: GaussianModel | None = None,
        orig_cam: bool = False,
    ):
        camera = client.camera
        w2c = get_w2c(camera)
        W = resolution
        H = int(resolution / camera.aspect)
        if W < 16 or H < 16:
            return
        if W > 4800 or H > 4800:
            return
        if orig_cam:
            fovY = self.cameras[0].fovY
            if self.cameras[0].image_width >= self.cameras[0].image_height:
                W = resolution
                H = int(
                    resolution
                    / (self.cameras[0].image_width / self.cameras[0].image_height)
                )
            else:
                H = resolution
                W = int(
                    resolution
                    / (self.cameras[0].image_height / self.cameras[0].image_width)
                )
        else:
            fovY = camera.fov
        fovX = focal2fov(fov2focal(fovY, H), W)
        R: torch.FloatTensor = torch.from_numpy(w2c[:3, :3]).transpose(1, 0)  # type: ignore
        T: torch.FloatTensor = torch.from_numpy(w2c[:3, 3] / VISER_SCALE)  # type: ignore
        world_view_transform = get_world2view(R, T).transpose(0, 1).cuda()
        znear = 0.01
        zfar = 1000.0
        projection_matrix = (
            get_projection_matrix(znear=znear, zfar=zfar, fovX=fovX, fovY=fovY)
            .transpose(0, 1)
            .cuda()
        )
        full_proj_transform = (
            world_view_transform.unsqueeze(0)
            .bmm(projection_matrix.unsqueeze(0))
            .squeeze(0)
            .cuda()
        )

        cam = ViewerCamera(
            width=W,
            height=H,
            fovX=fovX,
            fovY=fovY,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
        )

        set_gaussians = gaussians is not None
        mask_color = False
        selection = self.selection
        if not set_gaussians:
            gaussians = self.gaussians
            if (
                selection is not None
                and self.render_isolated_checkbox.value
                and (
                    not isinstance(selection, TextPromptSelection)
                    or self.similarity_map_checkbox is None
                    or not self.similarity_map_checkbox.value
                )
            ):
                gaussians = selection.gaussians
                mask_color = True

        override_color = None
        if self.labels is not None and (
            selection is None or not isinstance(selection, TextPromptSelection)
        ):
            if self.render_dropdown.value == "PCA":
                override_color = self.pca_colors
            elif (
                self.render_dropdown.value == "Labels"
                and self.render_isolated_checkbox.value
            ):
                override_color = self.cluster_colors
            elif not self.render_isolated_checkbox.value and selection is not None:
                if self.render_dropdown.value == "RGB":
                    override_color = SH2RGB(
                        gaussians.get_spherical_harmonics.detach().clone()[..., 0, :]
                    )
                else:
                    assert self.cluster_colors is not None, (
                        "Cluster colors are not set, cannot render without isolated mode"
                    )
                    override_color = self.cluster_colors.clone()
                override_color[~selection.mask] = 0
        elif selection is not None and isinstance(selection, TextPromptSelection):
            # language query
            if (
                self.similarity_map_checkbox is not None
                and self.similarity_map_checkbox.value
            ):
                override_color = selection.sim
            elif not self.render_isolated_checkbox.value:
                override_color = SH2RGB(
                    gaussians.get_spherical_harmonics.detach().clone()[..., 0, :]
                )
                override_color[~selection.mask] = 0

        if mask_color and override_color is not None and selection is not None:
            override_color = override_color[selection.mask]

        render_package = render(
            cam,
            gaussians,
            self.pipe,
            self.bg_colors["white" if self.bg_color_checkbox.value else "black"],
            self.setup_params.config.model.sh_degree,
            None,
            None,
            self.scale_slider.value,
            override_color=override_color,
            override_features=torch.Tensor([]).cuda() if not set_gaussians else None,
            render_color=self.render_dropdown.value != "Alpha",
            render_alpha=self.render_dropdown.value == "Alpha",
        )

        return render_package

    @torch.no_grad()
    def update(self) -> None:
        if not self.need_update:
            return
        resolution = self.resolution_slider.value
        orig_cam = False  # self.original_aspect_ratio_checkbox.value
        for client in self.server.get_clients().values():
            try:
                render_package = self.render(client, resolution, orig_cam=orig_cam)
                if render_package is not None:
                    if self.render_dropdown.value == "Alpha":
                        assert render_package.alpha is not None, "Alpha is None"
                        out = 1 - render_package.alpha.detach().cpu().numpy()
                        out = out * 255
                        out = np.stack([out, out, out], axis=-1).astype(np.uint8)
                    else:
                        assert render_package.render is not None, (
                            "Rendered image is None"
                        )
                        out = (
                            render_package.render.detach()
                            .permute(1, 2, 0)
                            .clip(0, 1)
                            .mul(255)
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )
                    if orig_cam:
                        if client.camera.aspect < 1:
                            W = resolution
                            H = int(resolution / client.camera.aspect)
                        else:
                            H = resolution
                            W = int(resolution * client.camera.aspect)
                        img = (
                            self.bg_colors[
                                "white" if self.bg_color_checkbox.value else "black"
                            ]
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .expand(H, W, 3)
                            .mul(255)
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )
                        half_h = (H - out.shape[0]) // 2
                        half_w = (W - out.shape[1]) // 2
                        img[
                            half_h : half_h + out.shape[0],
                            half_w : half_w + out.shape[1],
                        ] = out
                        out = img
                    client.scene.set_background_image(
                        out, format="jpeg", jpeg_quality=95
                    )
            except Exception as e:
                print("Exception:", e)
                print(traceback.format_exc())
                exit()
            # client.flush()  # Optional!
        time.sleep(1 / 60.0)
        self.need_update = False


def main(
    model_dir: str, vis_cameras: bool, language: str | None, resolution: int
) -> None:
    viewer = ViserViewer(model_dir, vis_cameras, language, resolution)

    while True:
        viewer.update()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to the model directory")
    parser.add_argument(
        "--language",
        type=str,
        choices=["clip", "siglip", "masqclip"],
        help="Language model",
    )
    parser.add_argument("--cameras", action="store_true", help="Visualize cameras")
    parser.add_argument(
        "--resolution", type=int, default=1240, help="Initial render resolution"
    )

    args = parser.parse_args()

    try:
        main(args.model_dir, args.cameras, args.language, args.resolution)
    except KeyboardInterrupt:
        print("\nExiting viewer...")
        exit(0)

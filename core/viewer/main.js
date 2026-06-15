// --- Extracted from splat/main.js ---

function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.2;
    const zfar = 200;
    return [
        (2 * fx) / width, 0, 0, 0,
        0, -(2 * fy) / height, 0, 0,
        0, 0, zfar / (zfar - znear), 1,
        0, 0, -(zfar * znear) / (zfar - znear), 0
    ];
}

function getViewMatrix(camera) {
    const R = camera.rotation.flat();
    const t = camera.position;
    const camToWorld = [
        [R[0], R[1], R[2], 0],
        [R[3], R[4], R[5], 0],
        [R[6], R[7], R[8], 0],
        [
            -t[0] * R[0] - t[1] * R[3] - t[2] * R[6],
            -t[0] * R[1] - t[1] * R[4] - t[2] * R[7],
            -t[0] * R[2] - t[1] * R[5] - t[2] * R[8],
            1,
        ],
    ].flat();
    return camToWorld;
}

function multiply4(a, b) {
    return [
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

function invert4(a) {
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    let det =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}

function rotate4(a, rad, x, y, z) {
    let len = Math.hypot(x, y, z);
    x /= len;
    y /= len;
    z /= len;
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    let t = 1 - c;
    let b00 = x * x * t + c;
    let b01 = y * x * t + z * s;
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c;
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s;
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        ...a.slice(12, 16),
    ];
}

function translate4(a, x, y, z) {
    return [
        ...a.slice(0, 12),
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}

function createWorker(self) {
    let buffer;
    let vertexCount = 0;
    let viewProj;
    // 6*4 + 4 + 4 = 8*4
    // XYZ - Position (Float32)
    // XYZ - Scale (Float32)
    // RGBA - colors (uint8)
    // IJKL - quaternion/rot (uint8)
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    let lastProj = [];
    let depthIndex = new Uint32Array();
    let lastVertexCount = 0;

    var _floatView = new Float32Array(1);
    var _int32View = new Int32Array(_floatView.buffer);

    function floatToHalf(float) {
        _floatView[0] = float;
        var f = _int32View[0];

        var sign = (f >> 31) & 0x0001;
        var exp = (f >> 23) & 0x00ff;
        var frac = f & 0x007fffff;

        var newExp;
        if (exp == 0) {
            newExp = 0;
        } else if (exp < 113) {
            newExp = 0;
            frac |= 0x00800000;
            frac = frac >> (113 - exp);
            if (frac & 0x01000000) {
                newExp = 1;
                frac = 0;
            }
        } else if (exp < 142) {
            newExp = exp - 112;
        } else {
            newExp = 31;
            frac = 0;
        }

        return (sign << 15) | (newExp << 10) | (frac >> 13);
    }

    function packHalf2x16(x, y) {
        return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
    }

    function generateTexture() {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        const u_buffer = new Uint8Array(buffer);

        var texwidth = 1024 * 2; // Set to your desired width
        var texheight = Math.ceil((2 * vertexCount) / texwidth); // Set to your desired height
        var texdata = new Uint32Array(texwidth * texheight * 4); // 4 components per pixel (RGBA)
        var texdata_c = new Uint8Array(texdata.buffer);
        var texdata_f = new Float32Array(texdata.buffer);

        // Here we convert from a .splat file buffer into a texture
        // With a little bit more foresight perhaps this texture file
        // should have been the native format as it'd be very easy to
        // load it into webgl.
        for (let i = 0; i < vertexCount; i++) {
            // x, y, z
            texdata_f[8 * i + 0] = f_buffer[8 * i + 0];
            texdata_f[8 * i + 1] = f_buffer[8 * i + 1];
            texdata_f[8 * i + 2] = f_buffer[8 * i + 2];

            // r, g, b, a
            texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
            texdata_c[4 * (8 * i + 7) + 1] = u_buffer[32 * i + 24 + 1];
            texdata_c[4 * (8 * i + 7) + 2] = u_buffer[32 * i + 24 + 2];
            texdata_c[4 * (8 * i + 7) + 3] = u_buffer[32 * i + 24 + 3];

            // quaternions
            let scale = [
                f_buffer[8 * i + 3 + 0],
                f_buffer[8 * i + 3 + 1],
                f_buffer[8 * i + 3 + 2],
            ];
            let rot = [
                (u_buffer[32 * i + 28 + 0] - 128) / 128,
                (u_buffer[32 * i + 28 + 1] - 128) / 128,
                (u_buffer[32 * i + 28 + 2] - 128) / 128,
                (u_buffer[32 * i + 28 + 3] - 128) / 128,
            ];

            // Compute the matrix product of S and R (M = S * R)
            const M = [
                1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
                2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
                2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),

                2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
                2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),

                2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
            ].map((k, i) => k * scale[Math.floor(i / 3)]);

            const sigma = [
                M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
                M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
                M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
                M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
                M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
                M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
            ];

            texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
            texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
            texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
        }

        self.postMessage({ texdata, texwidth, texheight }, [texdata.buffer]);
    }

    function runSort(viewProj) {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        if (lastVertexCount == vertexCount) {
            let dot =
                lastProj[2] * viewProj[2] +
                lastProj[6] * viewProj[6] +
                lastProj[10] * viewProj[10];
            if (Math.abs(dot - 1) < 0.01) {
                return;
            }
        } else {
            generateTexture();
            lastVertexCount = vertexCount;
        }

        console.time("sort");
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) {
            let depth =
                ((viewProj[2] * f_buffer[8 * i + 0] +
                    viewProj[6] * f_buffer[8 * i + 1] +
                    viewProj[10] * f_buffer[8 * i + 2]) *
                    4096) |
                0;
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }

        // This is a 16 bit single-pass counting sort
        let depthInv = (256 * 256 - 1) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);
        for (let i = 0; i < vertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++)
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        depthIndex = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++)
            depthIndex[starts0[sizeList[i]]++] = i;

        console.timeEnd("sort");

        lastProj = viewProj;
        self.postMessage({ depthIndex, viewProj, vertexCount }, [
            depthIndex.buffer,
        ]);
    }

    function processPlyBuffer(inputBuffer) {
        const ubuf = new Uint8Array(inputBuffer);
        // 10KB ought to be enough for a header...
        const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
        const header_end = "end_header\n";
        const header_end_index = header.indexOf(header_end);
        if (header_end_index < 0)
            throw new Error("Unable to read .ply file header");
        const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
        console.log("Vertex Count", vertexCount);
        let row_offset = 0,
            offsets = {},
            types = {};
        const TYPE_MAP = {
            double: "getFloat64",
            int: "getInt32",
            uint: "getUint32",
            float: "getFloat32",
            short: "getInt16",
            ushort: "getUint16",
            uchar: "getUint8",
        };
        for (let prop of header
            .slice(0, header_end_index)
            .split("\n")
            .filter((k) => k.startsWith("property "))) {
            const [p, type, name] = prop.split(" ");
            const arrayType = TYPE_MAP[type] || "getInt8";
            types[name] = arrayType;
            offsets[name] = row_offset;
            row_offset += parseInt(arrayType.replace(/[^\d]/g, "")) / 8;
        }
        console.log("Bytes per row", row_offset, types, offsets);

        let dataView = new DataView(
            inputBuffer,
            header_end_index + header_end.length,
        );
        let row = 0;
        const attrs = new Proxy(
            {},
            {
                get(target, prop) {
                    if (!types[prop]) throw new Error(prop + " not found");
                    return dataView[types[prop]](
                        row * row_offset + offsets[prop],
                        true,
                    );
                },
            },
        );

        console.time("calculate importance");
        let sizeList = new Float32Array(vertexCount);
        let sizeIndex = new Uint32Array(vertexCount);
        for (row = 0; row < vertexCount; row++) {
            sizeIndex[row] = row;
            if (!types["scale_0"]) continue;
            const size =
                Math.exp(attrs.scale_0) *
                Math.exp(attrs.scale_1) *
                Math.exp(attrs.scale_2);
            const opacity = 1 / (1 + Math.exp(-attrs.opacity));
            sizeList[row] = size * opacity;
        }
        console.timeEnd("calculate importance");

        console.time("sort");
        sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);
        console.timeEnd("sort");

        // 6*4 + 4 + 4 = 8*4
        // XYZ - Position (Float32)
        // XYZ - Scale (Float32)
        // RGBA - colors (uint8)
        // IJKL - quaternion/rot (uint8)
        const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
        const buffer = new ArrayBuffer(rowLength * vertexCount);

        console.time("build buffer");
        for (let j = 0; j < vertexCount; j++) {
            row = sizeIndex[j];

            const position = new Float32Array(buffer, j * rowLength, 3);
            const scales = new Float32Array(buffer, j * rowLength + 4 * 3, 3);
            const rgba = new Uint8ClampedArray(
                buffer,
                j * rowLength + 4 * 3 + 4 * 3,
                4,
            );
            const rot = new Uint8ClampedArray(
                buffer,
                j * rowLength + 4 * 3 + 4 * 3 + 4,
                4,
            );

            if (types["scale_0"]) {
                const qlen = Math.sqrt(
                    attrs.rot_0 ** 2 +
                    attrs.rot_1 ** 2 +
                    attrs.rot_2 ** 2 +
                    attrs.rot_3 ** 2,
                );

                rot[0] = (attrs.rot_0 / qlen) * 128 + 128;
                rot[1] = (attrs.rot_1 / qlen) * 128 + 128;
                rot[2] = (attrs.rot_2 / qlen) * 128 + 128;
                rot[3] = (attrs.rot_3 / qlen) * 128 + 128;

                scales[0] = Math.exp(attrs.scale_0);
                scales[1] = Math.exp(attrs.scale_1);
                scales[2] = Math.exp(attrs.scale_2);
            } else {
                scales[0] = 0.01;
                scales[1] = 0.01;
                scales[2] = 0.01;

                rot[0] = 255;
                rot[1] = 0;
                rot[2] = 0;
                rot[3] = 0;
            }

            position[0] = attrs.x;
            position[1] = attrs.y;
            position[2] = attrs.z;

            if (types["f_dc_0"]) {
                const SH_C0 = 0.28209479177387814;
                rgba[0] = (0.5 + SH_C0 * attrs.f_dc_0) * 255;
                rgba[1] = (0.5 + SH_C0 * attrs.f_dc_1) * 255;
                rgba[2] = (0.5 + SH_C0 * attrs.f_dc_2) * 255;
            } else {
                rgba[0] = attrs.red;
                rgba[1] = attrs.green;
                rgba[2] = attrs.blue;
            }
            if (types["opacity"]) {
                rgba[3] = (1 / (1 + Math.exp(-attrs.opacity))) * 255;
            } else {
                rgba[3] = 255;
            }
        }
        console.timeEnd("build buffer");
        return buffer;
    }

    const throttledSort = () => {
        if (!sortRunning) {
            sortRunning = true;
            let lastView = viewProj;
            runSort(lastView);
            setTimeout(() => {
                sortRunning = false;
                if (lastView !== viewProj) {
                    throttledSort();
                }
            }, 0);
        }
    };

    let sortRunning;
    self.onmessage = (e) => {
        if (e.data.ply) {
            vertexCount = 0;
            runSort(viewProj);
            buffer = processPlyBuffer(e.data.ply);
            vertexCount = Math.floor(buffer.byteLength / rowLength);
            postMessage({ buffer: buffer, save: !!e.data.save });
        } else if (e.data.buffer) {
            buffer = e.data.buffer;
            vertexCount = e.data.vertexCount;
            lastVertexCount = 0; // Force texture regeneration with new color/scale data
        } else if (e.data.vertexCount) {
            vertexCount = e.data.vertexCount;
        } else if (e.data.view) {
            viewProj = e.data.view;
            throttledSort();
        }
    };
}

const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;

void main () {
    uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
    vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1);
    vec4 pos2d = projection * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z),
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
    vPosition = position;

    vec2 vCenter = vec2(pos2d) / pos2d.w;
    gl_Position = vec4(
        vCenter
        + position.x * majorAxis / viewport
        + position.y * minorAxis / viewport, 0.0, 1.0);
}
`.trim();

const fragmentShaderSource = `
#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(B * vColor.rgb, B);
}
`.trim();

let defaultViewMatrix = [
    0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07,
    0.03, 6.55, 1,
];
let viewMatrix = [...defaultViewMatrix];
let keyframesList = [];
let lastInteractionTime = Date.now();

function mat4ToQuat(m) {
    let q = [0, 0, 0, 1]; // x, y, z, w
    let trace = m[0] + m[5] + m[10];
    if (trace > 0.0) {
        let s = Math.sqrt(trace + 1.0) * 2; 
        q[3] = 0.25 * s;
        q[0] = (m[6] - m[9]) / s;
        q[1] = (m[8] - m[2]) / s;
        q[2] = (m[1] - m[4]) / s;
    } else if ((m[0] > m[5]) && (m[0] > m[10])) {
        let s = Math.sqrt(1.0 + m[0] - m[5] - m[10]) * 2; 
        q[3] = (m[6] - m[9]) / s;
        q[0] = 0.25 * s;
        q[1] = (m[1] + m[4]) / s;
        q[2] = (m[8] + m[2]) / s;
    } else if (m[5] > m[10]) {
        let s = Math.sqrt(1.0 + m[5] - m[0] - m[10]) * 2; 
        q[3] = (m[8] - m[2]) / s;
        q[0] = (m[1] + m[4]) / s;
        q[1] = 0.25 * s;
        q[2] = (m[6] + m[9]) / s;
    } else {
        let s = Math.sqrt(1.0 + m[10] - m[0] - m[5]) * 2; 
        q[3] = (m[1] - m[4]) / s;
        q[0] = (m[8] + m[2]) / s;
        q[1] = (m[6] + m[9]) / s;
        q[2] = 0.25 * s;
    }
    return q;
}

function quatToMat4(q, pos) {
    let x = q[0], y = q[1], z = q[2], w = q[3];
    let x2 = x + x, y2 = y + y, z2 = z + z;
    let xx = x * x2, xy = x * y2, xz = x * z2;
    let yy = y * y2, yz = y * z2, zz = z * z2;
    let wx = w * x2, wy = w * y2, wz = w * z2;

    let res = new Array(16).fill(0);
    res[0] = 1.0 - (yy + zz);
    res[1] = xy + wz;
    res[2] = xz - wy;

    res[4] = xy - wz;
    res[5] = 1.0 - (xx + zz);
    res[6] = yz + wx;

    res[8] = xz + wy;
    res[9] = yz - wx;
    res[10] = 1.0 - (xx + yy);

    res[12] = pos[0];
    res[13] = pos[1];
    res[14] = pos[2];
    res[15] = 1.0;
    return res;
}

function slerpQuat(q1, q2, t) {
    let ax = q1[0], ay = q1[1], az = q1[2], aw = q1[3];
    let bx = q2[0], by = q2[1], bz = q2[2], bw = q2[3];

    let cosHalfTheta = ax * bx + ay * by + az * bz + aw * bw;

    if (cosHalfTheta < 0) {
        bx = -bx; by = -by; bz = -bz; bw = -bw;
        cosHalfTheta = -cosHalfTheta;
    }

    if (Math.abs(cosHalfTheta) >= 1.0) {
        return [ax, ay, az, aw];
    }

    let halfTheta = Math.acos(cosHalfTheta);
    let sinHalfTheta = Math.sqrt(1.0 - cosHalfTheta * cosHalfTheta);

    if (Math.abs(sinHalfTheta) < 0.001) {
        return [
            ax * 0.5 + bx * 0.5,
            ay * 0.5 + by * 0.5,
            az * 0.5 + bz * 0.5,
            aw * 0.5 + bw * 0.5
        ];
    }

    let ratioA = Math.sin((1 - t) * halfTheta) / sinHalfTheta;
    let ratioB = Math.sin(t * halfTheta) / sinHalfTheta;

    return [
        ax * ratioA + bx * ratioB,
        ay * ratioA + by * ratioB,
        az * ratioA + bz * ratioB,
        aw * ratioA + bw * ratioB
    ];
}

function interpolateMatrices(m1, m2, alpha) {
    const pos1 = [m1[12], m1[13], m1[14]];
    const pos2 = [m2[12], m2[13], m2[14]];
    
    const posInterp = [
        pos1[0] * (1.0 - alpha) + pos2[0] * alpha,
        pos1[1] * (1.0 - alpha) + pos2[1] * alpha,
        pos1[2] * (1.0 - alpha) + pos2[2] * alpha
    ];
    
    const q1 = mat4ToQuat(m1);
    const q2 = mat4ToQuat(m2);
    
    const qInterp = slerpQuat(q1, q2, alpha);
    
    return quatToMat4(qInterp, posInterp);
}

let labelsMetadata = [];
let bboxCanvas = null;
let bboxCtx = null;

function projectPoint(p, viewProj) {
    const x = p[0], y = p[1], z = p[2];
    const w = viewProj[3] * x + viewProj[7] * y + viewProj[11] * z + viewProj[15];
    if (w <= 0.2) return null; // Behind or too close to camera (w is positive in front of camera in our Z-forward system)

    const xp = viewProj[0] * x + viewProj[4] * y + viewProj[8] * z + viewProj[12];
    const yp = viewProj[1] * x + viewProj[5] * y + viewProj[9] * z + viewProj[13];

    const ndcX = xp / w;
    const ndcY = yp / w;

    return {
        x: (ndcX * 0.5 + 0.5) * window.innerWidth,
        y: (-ndcY * 0.5 + 0.5) * window.innerHeight
    };
}

function extractCameraVectors(c2w) {
    const position = [c2w[12], c2w[13], c2w[14]];
    // In our system c2w forward is typically +Z (index 8,9,10)
    const forward = [c2w[8], c2w[9], c2w[10]];
    const right = [c2w[0], c2w[1], c2w[2]];
    const up = [c2w[4], c2w[5], c2w[6]];
    const mag = Math.sqrt(forward[0]**2 + forward[1]**2 + forward[2]**2);
    return {
        position,
        forward: [forward[0]/mag, forward[1]/mag, forward[2]/mag],
        right,
        up
    };
}

function selectDiverseViews(objectCentroid, keyframes, k=4) {
    if (!keyframes || keyframes.length === 0) return [];
    
    // 1. Filter and compute V_pool
    const vPool = [];
    for (let i=0; i<keyframes.length; i++) {
        const cam = extractCameraVectors(keyframes[i]);
        
        const dv = [
            objectCentroid[0] - cam.position[0],
            objectCentroid[1] - cam.position[1],
            objectCentroid[2] - cam.position[2]
        ];
        const dist = Math.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2);
        if (dist === 0) continue;
        
        const dvNorm = [dv[0]/dist, dv[1]/dist, dv[2]/dist];
        const dotObj = cam.forward[0]*dvNorm[0] + cam.forward[1]*dvNorm[1] + cam.forward[2]*dvNorm[2];
        const angle = Math.acos(Math.max(-1, Math.min(1, dotObj)));
        
        // Visibility check: object must be in front of the camera (within ~45 deg FOV half-angle)
        if (angle < Math.PI / 4) {
            vPool.push({
                index: i,
                c2w: keyframes[i],
                cam: cam,
                angle: angle
            });
        }
    }
    
    if (vPool.length === 0) return [];
    
    // 2. Select initial view (most centered / min angle)
    vPool.sort((a, b) => a.angle - b.angle);
    const S = [vPool[0]];
    const unselected = vPool.slice(1);
    
    // 3. Iteratively maximize minimum angular disparity (aligned with descriptions.py)
    // using cosine_similarity between the view_dir (forward) of the cameras
    while (S.length < k && unselected.length > 0) {
        let minMaxSim = 1.0;
        let bestCandidateIdx = -1;
        
        for (let i=0; i<unselected.length; i++) {
            const cand = unselected[i];
            let maxSim = -1.0;
            
            for (let j=0; j<S.length; j++) {
                const s = S[j];
                // sim = cosine_similarity(cand.view_dir, s.view_dir)
                const sim = cand.cam.forward[0]*s.cam.forward[0] + cand.cam.forward[1]*s.cam.forward[1] + cand.cam.forward[2]*s.cam.forward[2];
                if (sim > maxSim) {
                    maxSim = sim;
                }
            }
            
            // We want to minimize the maximum similarity (which means maximizing disparity)
            if (maxSim < minMaxSim) {
                minMaxSim = maxSim;
                bestCandidateIdx = i;
            }
        }
        
        if (bestCandidateIdx !== -1) {
            S.push(unselected[bestCandidateIdx]);
            unselected.splice(bestCandidateIdx, 1);
        } else {
            break;
        }
    }
    
    return S;
}

async function main() {
    let carousel = false; // Set default carousel to false to align with real-time SLAM camera poses
    const params = new URLSearchParams(location.search);
    try {
        viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
    } catch (err) { }
    
    let serverConfig = {
        is_training: false,
        has_labels: false,
        has_bboxes: false
    };
    try {
        const configResp = await fetch('/config');
        serverConfig = await configResp.json();
    } catch (e) {
        console.warn("Could not fetch /config", e);
    }
    
    // Update UI based on config
    if (serverConfig.is_training) {
        document.getElementById('live-cameras-panel').style.display = 'block';
        document.getElementById('rendered-cameras-panel').style.display = 'block';
    } else {
        const toggleCamerasLabel = document.getElementById("toggle-cameras-label");
        if (toggleCamerasLabel) toggleCamerasLabel.style.display = "flex";
    }
    
    const modeSelect = document.getElementById("render-mode");
    if (modeSelect) {
        let optionsHtml = '<option value="rgb">RGB (Color Original)</option>';
        if (serverConfig.is_training) {
            optionsHtml += '<option value="semantic">Características (SAM2)</option>';
        }
        if (serverConfig.has_labels) {
            optionsHtml += '<option value="labels">Instancias</option>';
            optionsHtml += '<option value="rgb_labels">RGB + Etiquetas</option>';
        }
        if (serverConfig.has_bboxes) {
            optionsHtml += '<option value="rgb_labels_bboxes">RGB + Etiquetas + BBoxes</option>';
        }
        modeSelect.innerHTML = optionsHtml;
    }
    
    if (serverConfig.agent_mode) {
        const masterPanel = document.getElementById("master-panel");
        if (masterPanel) masterPanel.style.display = "none";

        const chatPanel = document.getElementById("chat-panel");
        if (chatPanel) {
            chatPanel.style.display = "flex";
            chatPanel.style.bottom = "auto";
            chatPanel.style.right = "auto";
            chatPanel.style.top = "15px";
            chatPanel.style.left = "15px";
        }
    }

    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    let splatData = new Uint8Array(0);
    let lastByteLength = 0;
    let isInitialLoad = true;
    let isInitialPoseSet = false;

    // Spawn sorting WebWorker
    const worker = new Worker(
        URL.createObjectURL(
            new Blob(["(", createWorker.toString(), ")(self)"], {
                type: "application/javascript",
            }),
        ),
    );

    const canvas = document.getElementById("canvas");
    bboxCanvas = document.getElementById("bbox-canvas");
    bboxCtx = bboxCanvas ? bboxCanvas.getContext("2d") : null;
    const fps = document.getElementById("fps");
    const camid = document.getElementById("camid");

    let projectionMatrix;

    const gl = canvas.getContext("webgl2", {
        antialias: false,
        preserveDrawingBuffer: true,
    });

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(vertexShader));

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(fragmentShader));

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    gl.disable(gl.DEPTH_TEST); // Disable depth testing
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
    );
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    const u_projection = gl.getUniformLocation(program, "projection");
    const u_viewport = gl.getUniformLocation(program, "viewport");
    const u_focal = gl.getUniformLocation(program, "focal");
    const u_view = gl.getUniformLocation(program, "view");
    const u_texture = gl.getUniformLocation(program, "u_texture");

    gl.uniform1i(u_texture, 0);

    const positionBuffer = gl.createBuffer();
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]),
        gl.STATIC_DRAW,
    );
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);

    const downsample = 1;

    const resize = () => {
        // Calculate vertical focal length (corresponds to camera vertical FOV)
        const fovRad = (60 * Math.PI) / 180;
        const fy = (innerHeight / 2.0) / Math.tan(fovRad / 2.0);
        const fx = fy;

        gl.uniform2fv(u_focal, new Float32Array([fx, fy]));

        projectionMatrix = getProjectionMatrix(
            fx,
            fy,
            innerWidth,
            innerHeight,
        );

        gl.uniform2fv(u_viewport, new Float32Array([innerWidth, innerHeight]));

        gl.canvas.width = Math.round(innerWidth / downsample);
        gl.canvas.height = Math.round(innerHeight / downsample);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

        gl.uniformMatrix4fv(u_projection, false, new Float32Array(projectionMatrix));

        if (bboxCanvas) {
            bboxCanvas.width = innerWidth;
            bboxCanvas.height = innerHeight;
        }
    };

    window.addEventListener("resize", resize);
    resize();

    let vertexCount = 0;

    worker.onmessage = (e) => {
        if (e.data.texdata) {
            const { texdata, texwidth, texheight } = e.data;
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA32UI,
                texwidth,
                texheight,
                0,
                gl.RGBA_INTEGER,
                gl.UNSIGNED_INT,
                texdata,
            );
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
        } else if (e.data.depthIndex) {
            const { depthIndex } = e.data;
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
            vertexCount = e.data.vertexCount;
        }
    };

    let activeKeys = [];

    window.addEventListener("keydown", (e) => {
        carousel = false;
        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
    });
    window.addEventListener("keyup", (e) => {
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });
    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    window.addEventListener(
        "wheel",
        (e) => {
            if (serverConfig.agent_mode) return;
            if (e.target.closest('#master-panel')) return;
            
            
            carousel = false;
            e.preventDefault();
            const lineHeight = 10;
            const scale = e.deltaMode == 1 ? lineHeight : e.deltaMode == 2 ? innerHeight : 1;
            let inv = invert4(viewMatrix);
            if (e.shiftKey) {
                inv = translate4(
                    inv,
                    (e.deltaX * scale) / innerWidth,
                    (e.deltaY * scale) / innerHeight,
                    0,
                );
            } else if (e.ctrlKey || e.metaKey) {
                inv = translate4(
                    inv,
                    0,
                    0,
                    (-10 * (e.deltaY * scale)) / innerHeight,
                );
            } else {
                let d = 4;
                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
                inv = rotate4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);
            }

            viewMatrix = invert4(inv);
        },
        { passive: false },
    );

    let startX, startY, down;
    canvas.addEventListener("mousedown", (e) => {
        if (serverConfig.agent_mode) return;
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = e.ctrlKey || e.metaKey ? 2 : 1;
    });
    canvas.addEventListener("contextmenu", (e) => {
        if (serverConfig.agent_mode) return;
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = 2;
    });

    canvas.addEventListener("mousemove", (e) => {
        e.preventDefault();
        
        

        if (down == 1) {
            let inv = invert4(viewMatrix);
            let dx = (5 * (e.clientX - startX)) / innerWidth;
            let dy = (5 * (e.clientY - startY)) / innerHeight;
            let d = 4;

            inv = translate4(inv, 0, 0, d);
            inv = rotate4(inv, dx, 0, 1, 0);
            inv = rotate4(inv, -dy, 1, 0, 0);
            inv = translate4(inv, 0, 0, -d);
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        } else if (down == 2) {
            let inv = invert4(viewMatrix);
            inv = translate4(
                inv,
                (-10 * (e.clientX - startX)) / innerWidth,
                0,
                (10 * (e.clientY - startY)) / innerHeight,
            );
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        }
    });
    canvas.addEventListener("mouseup", (e) => {
        e.preventDefault();
        down = false;
        startX = 0;
        startY = 0;
    });

    let altX = 0,
        altY = 0;
    canvas.addEventListener(
        "touchstart",
        (e) => {
            if (serverConfig.agent_mode) return;
            e.preventDefault();
            if (e.touches.length === 1) {
                carousel = false;
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                down = 1;
            } else if (e.touches.length === 2) {
                carousel = false;
                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
                down = 1;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchmove",
        (e) => {
            if (serverConfig.agent_mode) return;
            e.preventDefault();
            
            

            if (e.touches.length === 1 && down) {
                let inv = invert4(viewMatrix);
                let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
                let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

                let d = 4;
                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, dx, 0, 1, 0);
                inv = rotate4(inv, -dy, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                const dtheta =
                    Math.atan2(startY - altY, startX - altX) -
                    Math.atan2(
                        e.touches[0].clientY - e.touches[1].clientY,
                        e.touches[0].clientX - e.touches[1].clientX,
                    );
                const dscale =
                    Math.hypot(startX - altX, startY - altY) /
                    Math.hypot(
                        e.touches[0].clientX - e.touches[1].clientX,
                        e.touches[0].clientY - e.touches[1].clientY,
                    );
                const dx =
                    (e.touches[0].clientX +
                        e.touches[1].clientX -
                        (startX + altX)) /
                    2;
                const dy =
                    (e.touches[0].clientY +
                        e.touches[1].clientY -
                        (startY + altY)) /
                    2;
                let inv = invert4(viewMatrix);
                inv = rotate4(inv, dtheta, 0, 0, 1);

                inv = translate4(inv, -dx / innerWidth, -dy / innerHeight, 0);
                inv = translate4(inv, 0, 0, 3 * (1 - dscale));

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchend",
        (e) => {
            e.preventDefault();
            down = false;
            startX = 0;
            startY = 0;
        },
        { passive: false },
    );

    const resetIdle = () => { lastInteractionTime = Date.now(); };
    window.addEventListener("keydown", resetIdle);
    window.addEventListener("keyup", resetIdle);
    window.addEventListener("wheel", resetIdle, { passive: true });
    canvas.addEventListener("mousedown", resetIdle);
    canvas.addEventListener("mousemove", resetIdle);
    canvas.addEventListener("mouseup", resetIdle);
    canvas.addEventListener("touchstart", resetIdle, { passive: true });
    canvas.addEventListener("touchmove", resetIdle, { passive: true });
    canvas.addEventListener("touchend", resetIdle, { passive: true });

    let jumpDelta = 0;
    let lastFrame = 0;
    let avgFps = 0;
    let start = Date.now();

    // Semantic Navigation Chat State
    let chatFlightTarget = null;
    let chatFlightStart = null;
    let chatFlightStartTime = 0;
    const CHAT_FLIGHT_DURATION = 2500.0;

    const frame = (now) => {
        
        

        let inv = invert4(viewMatrix);
        let shiftKey =
            activeKeys.includes("Shift") ||
            activeKeys.includes("ShiftLeft") ||
            activeKeys.includes("ShiftRight");

        if (!serverConfig.agent_mode) {
            if (activeKeys.includes("ArrowUp")) {
            if (shiftKey) {
                inv = translate4(inv, 0, -0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, 0.1);
            }
        }
        if (activeKeys.includes("ArrowDown")) {
            if (shiftKey) {
                inv = translate4(inv, 0, 0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, -0.1);
            }
        }
        if (activeKeys.includes("ArrowLeft"))
            inv = translate4(inv, -0.03, 0, 0);
        if (activeKeys.includes("ArrowRight"))
            inv = translate4(inv, 0.03, 0, 0);

        if (activeKeys.includes("KeyA")) inv = rotate4(inv, -0.01, 0, 1, 0);
        if (activeKeys.includes("KeyD")) inv = rotate4(inv, 0.01, 0, 1, 0);
        if (activeKeys.includes("KeyQ")) inv = rotate4(inv, 0.01, 0, 0, 1);
        if (activeKeys.includes("KeyE")) inv = rotate4(inv, -0.01, 0, 0, 1);
        if (activeKeys.includes("KeyW")) inv = rotate4(inv, 0.005, 1, 0, 0);
        if (activeKeys.includes("KeyS")) inv = rotate4(inv, -0.005, 1, 0, 0);

        if (
            ["KeyJ", "KeyK", "KeyL", "KeyI"].some((k) => activeKeys.includes(k))
        ) {
            let d = 4;
            inv = translate4(inv, 0, 0, d);
            inv = rotate4(
                inv,
                activeKeys.includes("KeyJ")
                    ? -0.05
                    : activeKeys.includes("KeyL")
                        ? 0.05
                        : 0,
                0,
                1,
                0,
            );
            inv = rotate4(
                inv,
                activeKeys.includes("KeyI")
                    ? 0.05
                    : activeKeys.includes("KeyK")
                        ? -0.05
                        : 0,
                1,
                0,
                0,
            );
            inv = translate4(inv, 0, 0, -d);
        }

        } // End of !agent_mode keyboard check

        viewMatrix = invert4(inv);

        if (carousel) {
            let inv = invert4(defaultViewMatrix);
            const t = Math.sin((Date.now() - start) / 5000);
            inv = translate4(inv, 2.5 * t, 0, 6 * (1 - Math.cos(t)));
            inv = rotate4(inv, -0.6 * t, 0, 1, 0);
            viewMatrix = invert4(inv);
        }
        
        // --- CHAT FLIGHT ANIMATION ---
        if (chatFlightTarget && chatFlightStart) {
            const elapsed = Date.now() - chatFlightStartTime;
            let alpha = Math.min(elapsed / CHAT_FLIGHT_DURATION, 1.0);
            const easeAlpha = (1.0 - Math.cos(alpha * Math.PI)) / 2.0; // Smooth transition
            
            const c2wInterp = interpolateMatrices(chatFlightStart, chatFlightTarget, easeAlpha);
            const w2c = invert4(c2wInterp);
            if (w2c) {
                viewMatrix = w2c;
            }
            
            // End flight
            if (alpha >= 1.0) {
                chatFlightStart = null;
                chatFlightTarget = null;
            }
        }

        let actualViewMatrix = viewMatrix;

        const viewProj = multiply4(projectionMatrix, actualViewMatrix);
        worker.postMessage({ view: new Float32Array(viewProj) });

        // Update floating 3D labels dynamically based on camera projection
        const container = document.getElementById("labels-container");
        const modeSelect = document.getElementById("render-mode");
        const objSelect = document.getElementById("object-select");
        const activeMode = modeSelect ? modeSelect.value : "rgb";
        const activeFilter = objSelect ? objSelect.value : "all";

        if (bboxCtx && bboxCanvas) {
            bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
        }

        if (labelsMetadata && labelsMetadata.length > 0 && container) {
            labelsMetadata.forEach(lbl => {
                if (lbl.element) {
                    const isGenericObject = lbl.name && /^(objeto|objeto\s+|object|object\s+|Objeto|Objeto\s+)#\d+/i.test(lbl.name);
                    const isModeLabel = (activeMode === "rgb_labels" || activeMode === "rgb_labels_bboxes");
                    // Show if we are in a label mode OR if this specific object was selected by the VLM (activeFilter)
                    const shouldShow = ((isModeLabel && (activeFilter === "all" || activeFilter == lbl.id)) || 
                                        (activeFilter != "all" && activeFilter == lbl.id)) && 
                                       !isGenericObject;
                    if (shouldShow && lbl.center) {
                        const proj = projectPoint(lbl.center, viewProj);
                        if (proj) {
                            // Centroid placement for labels as requested: "las etiquetas déjalas en mitad de los objetos"
                            lbl.element.style.display = "block";
                            lbl.element.style.left = `${proj.x}px`;
                            lbl.element.style.top = `${proj.y}px`;

                            // True 3D perspective wireframe bounding box rendering as requested: "las bounding boxes me gustaría que fueran tridimensionales"
                            if (lbl.bbox && bboxCtx && activeMode === "rgb_labels_bboxes") {
                                const b = lbl.bbox;
                                const corners = [
                                    [b.min[0], b.min[1], b.min[2]], // 0
                                    [b.max[0], b.min[1], b.min[2]], // 1
                                    [b.max[0], b.max[1], b.min[2]], // 2
                                    [b.min[0], b.max[1], b.min[2]], // 3
                                    [b.min[0], b.min[1], b.max[2]], // 4
                                    [b.max[0], b.min[1], b.max[2]], // 5
                                    [b.max[0], b.max[1], b.max[2]], // 6
                                    [b.min[0], b.max[1], b.max[2]]  // 7
                                ];

                                const projected = corners.map(c => projectPoint(c, viewProj));

                                const edges = [
                                    [0, 1], [1, 2], [2, 3], [3, 0], // Bottom face
                                    [4, 5], [5, 6], [6, 7], [7, 4], // Top face
                                    [0, 4], [1, 5], [2, 6], [3, 7]  // Vertical edges
                                ];

                                bboxCtx.strokeStyle = `rgba(${lbl.color[0]}, ${lbl.color[1]}, ${lbl.color[2]}, 1.0)`; // Bold solid color (more intense)
                                bboxCtx.lineWidth = 3.5; // Thicker bounding box lines as requested
                                bboxCtx.setLineDash([6, 5]); // Adjusted dashed pattern to look clean at larger thickness

                                edges.forEach(edge => {
                                    const p1 = projected[edge[0]];
                                    const p2 = projected[edge[1]];
                                    if (p1 && p2) {
                                        bboxCtx.beginPath();
                                        bboxCtx.moveTo(p1.x, p1.y);
                                        bboxCtx.lineTo(p2.x, p2.y);
                                        bboxCtx.stroke();
                                    }
                                });
                            }
                        } else {
                            lbl.element.style.display = "none";
                        }
                    } else {
                        lbl.element.style.display = "none";
                    }
                }
            });
        }

        // Calculate and draw cameras if an object is selected and mode allows
        let selectedCameras = [];
        let activeObjCentroid = null;
        let activeObjColor = [0, 230, 118];
        
        if (labelsMetadata && activeFilter !== "all") {
            const activeLbl = labelsMetadata.find(l => l.id == activeFilter);
            if (activeLbl && activeLbl.center) {
                activeObjCentroid = activeLbl.center;
                activeObjColor = activeLbl.color;
                // Run Geometric Diversity View Selection (K=3)
                if (keyframesList) {
                    selectedCameras = selectDiverseViews(activeObjCentroid, keyframesList, 3);
                }
            }
        }

        // Check camera toggle
        const toggleCamerasEl = document.getElementById("toggle-cameras");
        const showCameras = toggleCamerasEl ? toggleCamerasEl.checked : false;

        // Render cameras
        if (showCameras && bboxCtx && keyframesList && (activeMode === "rgb_labels" || activeMode === "rgb_labels_bboxes")) {
            // Draw cameras
            keyframesList.forEach((c2w, i) => {
                const isSelected = selectedCameras.find(c => c.index === i);
                const cam = extractCameraVectors(c2w);
                
                // Camera center
                const p0 = cam.position;
                const projP0 = projectPoint(p0, viewProj);
                
                if (projP0) {
                    if (!isSelected) {
                        // Draw simple dots for unselected cameras
                        bboxCtx.beginPath();
                        bboxCtx.arc(projP0.x, projP0.y, 3, 0, 2 * Math.PI);
                        bboxCtx.fillStyle = 'rgba(255, 255, 255, 0.2)';
                        bboxCtx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
                        bboxCtx.lineWidth = 1;
                        bboxCtx.fill();
                        bboxCtx.stroke();
                    } else {
                        // Draw Frustum for selected cameras
                        const f = 0.2; // Smaller frustum size as requested
                        const w = 0.6 * f; // Width ratio
                        const h = 0.45 * f; // Height ratio
                        
                        // Center of the base plane
                        const centerPlane = [
                            p0[0] + cam.forward[0]*f,
                            p0[1] + cam.forward[1]*f,
                            p0[2] + cam.forward[2]*f
                        ];
                        
                        const corners = [
                            // Top-Left
                            [centerPlane[0] - cam.right[0]*w + cam.up[0]*h, centerPlane[1] - cam.right[1]*w + cam.up[1]*h, centerPlane[2] - cam.right[2]*w + cam.up[2]*h],
                            // Top-Right
                            [centerPlane[0] + cam.right[0]*w + cam.up[0]*h, centerPlane[1] + cam.right[1]*w + cam.up[1]*h, centerPlane[2] + cam.right[2]*w + cam.up[2]*h],
                            // Bottom-Right
                            [centerPlane[0] + cam.right[0]*w - cam.up[0]*h, centerPlane[1] + cam.right[1]*w - cam.up[1]*h, centerPlane[2] + cam.right[2]*w - cam.up[2]*h],
                            // Bottom-Left
                            [centerPlane[0] - cam.right[0]*w - cam.up[0]*h, centerPlane[1] - cam.right[1]*w - cam.up[1]*h, centerPlane[2] - cam.right[2]*w - cam.up[2]*h],
                        ];
                        
                        const projCorners = corners.map(c => projectPoint(c, viewProj));
                        
                        if (projCorners.every(p => p !== null)) {
                            // Set style for selected (more marked/bold)
                            bboxCtx.strokeStyle = `rgba(${activeObjColor[0]}, ${activeObjColor[1]}, ${activeObjColor[2]}, 1.0)`;
                            bboxCtx.lineWidth = 3.5; // Bolder lines
                            bboxCtx.fillStyle = `rgba(${activeObjColor[0]}, ${activeObjColor[1]}, ${activeObjColor[2]}, 0.45)`;

                            // Draw Frustum base (rectangle)
                            bboxCtx.beginPath();
                            bboxCtx.moveTo(projCorners[0].x, projCorners[0].y);
                            bboxCtx.lineTo(projCorners[1].x, projCorners[1].y);
                            bboxCtx.lineTo(projCorners[2].x, projCorners[2].y);
                            bboxCtx.lineTo(projCorners[3].x, projCorners[3].y);
                            bboxCtx.closePath();
                            bboxCtx.fill();
                            bboxCtx.stroke();
                            
                            // Draw lines from apex to base
                            for(let c of projCorners) {
                                bboxCtx.beginPath();
                                bboxCtx.moveTo(projP0.x, projP0.y);
                                bboxCtx.lineTo(c.x, c.y);
                                bboxCtx.stroke();
                            }
                            
                            // Draw apex point
                            bboxCtx.beginPath();
                            bboxCtx.arc(projP0.x, projP0.y, 6, 0, 2 * Math.PI);
                            bboxCtx.fillStyle = `rgba(${activeObjColor[0]}, ${activeObjColor[1]}, ${activeObjColor[2]}, 1.0)`;
                            bboxCtx.fill();
                            bboxCtx.strokeStyle = 'white';
                            bboxCtx.lineWidth = 2;
                            bboxCtx.stroke();
                            
                            // Draw connecting line to object
                            if (activeObjCentroid) {
                                const objProj = projectPoint(activeObjCentroid, viewProj);
                                if (objProj) {
                                    bboxCtx.beginPath();
                                    bboxCtx.moveTo(projP0.x, projP0.y);
                                    bboxCtx.lineTo(objProj.x, objProj.y);
                                    bboxCtx.strokeStyle = `rgba(${activeObjColor[0]}, ${activeObjColor[1]}, ${activeObjColor[2]}, 0.9)`;
                                    bboxCtx.lineWidth = 2.5;
                                    bboxCtx.setLineDash([5, 5]);
                                    bboxCtx.stroke();
                                    bboxCtx.setLineDash([]); // Reset dash
                                }
                            }
                        }
                    }
                }
            });
        }

        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;

        if (vertexCount > 0) {
            document.getElementById("spinner").style.display = "none";
            gl.uniformMatrix4fv(u_view, false, new Float32Array(actualViewMatrix));
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            document.getElementById("spinner").style.display = "";
            start = Date.now() + 2000;
        }

        fps.innerText = Math.round(avgFps) + " fps";
        lastFrame = now;
        requestAnimationFrame(frame);
    };

    frame();

    let lastUrl = "/gaussians";

    let masterCollapsed = false;
    window.toggleMasterPanel = function () {
        masterCollapsed = !masterCollapsed;
        const content = document.getElementById("master-content");
        const arrow = document.getElementById("master-toggle-arrow");

        if (masterCollapsed) {
            content.style.maxHeight = "0";
            content.style.opacity = "0";
            content.style.marginTop = "0";
            arrow.style.transform = "rotate(-90deg)";
            document.getElementById("live-rgb").src = "";
            document.getElementById("live-depth").src = "";
            const liveSam = document.getElementById("live-sam");
            if (liveSam) liveSam.src = "";
            const renderedRgb = document.getElementById("rendered-rgb");
            if (renderedRgb) renderedRgb.src = "";
            const renderedDepth = document.getElementById("rendered-depth");
            if (renderedDepth) renderedDepth.src = "";
        } else {
            content.style.maxHeight = "80vh";
            content.style.opacity = "1";
            content.style.marginTop = "10px";
            arrow.style.transform = "rotate(0deg)";
        }
    };

    // --- Dynamic live polling from SLAM backend endpoints ---
    async function pollData() {
        try {
            // Poll live RGB and Depth images if the panel is expanded
            if (!masterCollapsed) {
                const timestamp = Date.now();
                const liveRgbImg = document.getElementById("live-rgb");
                const liveDepthImg = document.getElementById("live-depth");
                const renderedRgbImg = document.getElementById("rendered-rgb");
                const renderedDepthImg = document.getElementById("rendered-depth");
                if (liveRgbImg) liveRgbImg.src = `/live_image?t=${timestamp}`;
                if (liveDepthImg) liveDepthImg.src = `/live_depth?t=${timestamp}`;
                if (renderedRgbImg) renderedRgbImg.src = `/rendered_image?t=${timestamp}`;
                if (renderedDepthImg) renderedDepthImg.src = `/rendered_depth?t=${timestamp}`;
            }

            // Poll cameras/poses to auto-follow tracker if user is not actively navigating
            const resPoses = await fetch("/poses");
            if (resPoses.ok) {
                const posesData = await resPoses.json();
                if (posesData.keyframes && posesData.keyframes.length > 0) {
                    keyframesList = posesData.keyframes;
                    if (!isInitialPoseSet) {
                        viewMatrix = invert4(keyframesList[0]);
                        isInitialPoseSet = true;
                    }
                }
                const currentPose = posesData.current_pose;

                const cameraStatusEl = document.getElementById("camera-status");
                if (cameraStatusEl) {
                    cameraStatusEl.textContent = "Navegación manual libre.";
                    cameraStatusEl.style.color = "#ff4a4a";
                }
            }

            // Determine query URL dynamically based on active mode/object selection
            let url = "/gaussians";
            const modeSelect = document.getElementById("render-mode");
            const objSelect = document.getElementById("object-select");
            const toggleCamerasEl = document.getElementById("toggle-cameras");
            const showCameras = toggleCamerasEl ? toggleCamerasEl.checked : false;

            if (modeSelect) {
                const mode = modeSelect.value;
                // If cameras are shown, force backend to send all data to preserve background
                let id = objSelect ? objSelect.value : "all";
                if (showCameras && id !== "all") {
                    id = "all";
                }
                url += `?mode=${mode}&id=${id}`;
            }

            // Poll binary Gaussian splats
            const resGaussians = await fetch(url);
            if (resGaussians.ok) {
                const arrayBuffer = await resGaussians.arrayBuffer();
                const byteLength = arrayBuffer.byteLength;

                if (byteLength > 0 && (byteLength !== lastByteLength || url !== lastUrl)) {
                    lastByteLength = byteLength;
                    lastUrl = url;
                    splatData = new Uint8Array(arrayBuffer);

                    const countEl = document.getElementById("gaussian-count");
                    const numGaussians = Math.floor(splatData.length / rowLength);
                    if (countEl) {
                        countEl.textContent = numGaussians.toLocaleString();
                    }

                    worker.postMessage({
                        buffer: splatData.buffer,
                        vertexCount: numGaussians,
                    });

                    // Focus camera on first Gaussian coordinates on initial non-zero load!
                    if (isInitialLoad) {
                        isInitialLoad = false;
                        if (!isInitialPoseSet) {
                            const floatView = new Float32Array(arrayBuffer, 0, 3);
                        const x = floatView[0];
                        const y = floatView[1];
                        const z = floatView[2];

                        let inv = invert4(defaultViewMatrix);
                        inv[12] = x;
                        inv[13] = y + 1.0;
                            inv[14] = z + 4.0;
                            viewMatrix = invert4(inv);
                        }
                    }
                }
            }
        } catch (e) {
            console.error("Polling error:", e);
        }

        setTimeout(pollData, 500);
    }

    // --- Fetch and populate semantic dropdowns if static demo/semantic session is active ---
    async function initSemanticPanel() {
        try {
            const res = await fetch("/labels");
            if (res.ok) {
                const data = await res.json();
                const panel = document.getElementById("master-panel");
                const modeSelect = document.getElementById("render-mode");
                const objSelect = document.getElementById("object-select");

                if (panel && modeSelect && objSelect && data.labels && data.labels.length > 0) {
                    // Panel is always block now, but we enable the dropdowns
                    objSelect.disabled = false;
                    labelsMetadata = data.labels;

                    const container = document.getElementById("labels-container");

                    // Populate dropdown
                    data.labels.forEach(lbl => {
                        const opt = document.createElement("option");
                        opt.value = lbl.id;
                        opt.textContent = lbl.name;
                        objSelect.appendChild(opt);

                        // Create a floating text box for each labeled object (excluding unlabeled/background -1)
                        if (lbl.id !== -1 && lbl.center && container) {
                            const div = document.createElement("div");
                            div.id = `floating-label-${lbl.id}`;
                            const cleanName = lbl.name.split(": ")[1] || lbl.name;
                            div.textContent = cleanName;

                            // High glassmorphic styling
                            div.style.cssText = `
                                position: absolute;
                                display: none;
                                background: rgba(0, 0, 0, 0.55);
                                backdrop-filter: blur(4px);
                                color: white;
                                padding: 6px 12px;
                                border-radius: 8px;
                                border: 2.5px solid rgba(${lbl.color[0]}, ${lbl.color[1]}, ${lbl.color[2]}, 0.7);
                                font-size: 25px;
                                font-weight: bold;
                                font-family: system-ui, sans-serif;
                                white-space: nowrap;
                                transform: translate(-50%, -50%);
                                pointer-events: none;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                                text-shadow: 0 0 3px black;
                                z-index: 510;
                            `;
                            container.appendChild(div);
                            lbl.element = div;
                        }
                    });

                    // Enable/disable select based on mode
                    modeSelect.addEventListener("change", () => {
                        if (modeSelect.value === "labels" || modeSelect.value === "rgb_labels") {
                            objSelect.disabled = false;
                        } else {
                            objSelect.disabled = true;
                            objSelect.value = "all";
                        }
                        // Reset lastByteLength and lastUrl to force instant reload
                        lastByteLength = 0;
                        lastUrl = "";
                    });

                    objSelect.addEventListener("change", () => {
                        // Reset lastByteLength and lastUrl to force instant reload
                        lastByteLength = 0;
                        lastUrl = "";
                        const info = document.getElementById("object-info");
                        if (info) {
                            const selectedOpt = objSelect.options[objSelect.selectedIndex];
                            info.textContent = selectedOpt.value !== "all" ? `Filtrando: ${selectedOpt.textContent}` : "";
                        }
                    });
                }
            }
        } catch (e) {
            console.log("No semantic session /labels endpoint found.");
        }
    }

    // --- SEMANTIC CHAT NAVIGATION LOGIC ---
    function setupChatNavigation() {
        const input = document.getElementById("chat-input");
        const sendBtn = document.getElementById("chat-send");
        const messages = document.getElementById("chat-messages");
        
        if (!input || !sendBtn || !messages) return;

        function addMessage(sender, text, isSystem = false) {
            const div = document.createElement("div");
            div.style.background = isSystem ? "rgba(255, 255, 255, 0.05)" : "rgba(255,255,255,0.1)";
            div.style.padding = "8px 10px";
            div.style.borderRadius = "6px";
            div.style.alignSelf = isSystem ? "flex-start" : "flex-end";
            div.style.maxWidth = "85%";
            div.style.border = isSystem ? "1px solid rgba(255, 255, 255, 0.1)" : "none";
            
            const span = document.createElement("span");
            span.style.color = isSystem ? "#ccc" : "#aaa";
            span.style.fontSize = "10px";
            span.style.display = "block";
            span.style.marginBottom = "2px";
            span.innerText = sender;
            
            div.appendChild(span);
            div.appendChild(document.createTextNode(text));
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }

        async function handleChat() {
            const text = input.value.trim();
            if (!text) return;
            
            addMessage("Usuario", text);
            input.value = "";
            
            
            

            if (typeof labelsMetadata === 'undefined' || !labelsMetadata) {
                addMessage("Sistema", "Todavía estoy cargando el mapa de objetos del entorno. Dame un segundo.", true);
                return;
            }

            const availableLabels = labelsMetadata.filter(l => l.id !== -1).map(l => {
                return { name: l.name, description: l.description || "Sin descripción adicional" };
            });
            addMessage("Sistema", "Analizando el entorno y tu petición...", true);
            
            // Extract current frame as base64 JPEG to help the LLM decide
            const imgData = canvas.toDataURL("image/jpeg", 0.5);
            
            let targetName = null;
            try {
                const response = await fetch("/reason", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        query: text,
                        available_labels: availableLabels,
                        image: imgData
                    })
                });
                
                if (!response.ok) throw new Error("Error HTTP " + response.status);
                const data = await response.json();
                targetName = data.target;
            } catch (err) {
                console.error(err);
                addMessage("Sistema", "Parece que hay un problema de conexión con mi cerebro artificial (Error 8081).", true);
                return;
            }

            if (!targetName || targetName === "UNKNOWN") {
                addMessage("Sistema", "No entiendo muy bien a dónde quieres ir, ¿me lo repites?", true);
                return;
            }

            let bestMatch = labelsMetadata.find(lbl => lbl.name === targetName);
            if (!bestMatch) {
                // Fallback in case LLM gave partial output
                let cleanTarget = targetName.replace(/\s+/g, '').toLowerCase();
                bestMatch = labelsMetadata.find(lbl => {
                    if (lbl.id === -1) return false;
                    let cleanLbl = lbl.name.replace(/\s+/g, '').toLowerCase();
                    return cleanLbl.includes(cleanTarget) || cleanTarget.includes(cleanLbl) || 
                           (cleanTarget.includes("#" + lbl.id) || cleanTarget === lbl.id.toString());
                });
            }

            if (bestMatch && bestMatch.center) {
                const cleanName = bestMatch.name.split(": ")[1] || bestMatch.name;
                addMessage("Sistema", `Vamos hacia allí. Poniendo rumbo a: ${cleanName}...`, true);
                
                const currentC2W = invert4(viewMatrix);
                
                // First try to find the absolute best training camera view that looks at this object
                let bestViewCamera = null;
                if (typeof keyframesList !== 'undefined' && keyframesList && keyframesList.length > 0) {
                    const views = selectDiverseViews(bestMatch.center, keyframesList, 1);
                    if (views && views.length > 0) {
                        bestViewCamera = views[0].c2w;
                    }
                }
                
                if (bestViewCamera) {
                    chatFlightTarget = bestViewCamera;
                } else {
                    // Fallback to manual 3D LookAt flight
                    const camPos = [currentC2W[12], currentC2W[13], currentC2W[14]];
                    const objPos = bestMatch.center;
                    
                    // 1. Calculate 3D direction towards the center
                    let fwd = [objPos[0]-camPos[0], objPos[1]-camPos[1], objPos[2]-camPos[2]];
                    let dist = Math.sqrt(fwd[0]**2 + fwd[1]**2 + fwd[2]**2);
                    fwd = dist > 0 ? [fwd[0]/dist, fwd[1]/dist, fwd[2]/dist] : [0, 0, 1];
                    
                    // 2. Target position: stop 1.2m away from the object in 3D
                    const offsetDist = 1.2;
                    let targetPos = camPos.slice();
                    if (dist > offsetDist) {
                        targetPos = [
                            objPos[0] - fwd[0]*offsetDist,
                            objPos[1] - fwd[1]*offsetDist,
                            objPos[2] - fwd[2]*offsetDist
                        ];
                    }
                    
                    // 3. Look perfectly at the object, maintaining the camera's current roll
                    let currentUp = [currentC2W[4], currentC2W[5], currentC2W[6]];
                    
                    // Right = currentUp x fwd (to maintain right-handed coordinate system)
                    let right = [
                        currentUp[1]*fwd[2] - currentUp[2]*fwd[1],
                        currentUp[2]*fwd[0] - currentUp[0]*fwd[2],
                        currentUp[0]*fwd[1] - currentUp[1]*fwd[0]
                    ];
                    let rightMag = Math.sqrt(right[0]**2 + right[1]**2 + right[2]**2);
                    if (rightMag < 0.001) {
                        right = [1, 0, 0];
                        rightMag = 1.0;
                    }
                    right = [right[0]/rightMag, right[1]/rightMag, right[2]/rightMag];
                    
                    // True Up = fwd x right (to guarantee orthogonality)
                    let up = [
                        fwd[1]*right[2] - fwd[2]*right[1],
                        fwd[2]*right[0] - fwd[0]*right[2],
                        fwd[0]*right[1] - fwd[1]*right[0]
                    ];

                    chatFlightTarget = [
                        right[0], right[1], right[2], 0,
                        up[0],    up[1],    up[2],    0,
                        fwd[0],   fwd[1],   fwd[2],   0,
                        targetPos[0], targetPos[1], targetPos[2], 1
                    ];
                }
                
                chatFlightStart = currentC2W;
                chatFlightStartTime = Date.now();
                
                // Select object automatically
                const objSelect = document.getElementById("object-select");
                if (objSelect && !objSelect.disabled) {
                    objSelect.value = bestMatch.id;
                    objSelect.dispatchEvent(new Event("change"));
                }
                
            } else {
                addMessage("Sistema", "No he podido encontrar ese objeto a mi alrededor, ¿puedes ser más específico?", true);
            }
        }

        sendBtn.addEventListener("click", handleChat);
        input.addEventListener("keypress", (e) => {
            if (e.key === "Enter") handleChat();
        });
    }

    setupChatNavigation();
    initSemanticPanel();
    pollData();
}

main().catch((err) => {
    document.getElementById("spinner").style.display = "none";
    document.getElementById("message").innerText = err.toString();
});

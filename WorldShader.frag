#version 130

uniform sampler2D texture;
uniform bool colorMapEnabled;
uniform mat3 colorMatrix;

void main() {
    vec4 col = texture2D(texture, gl_TexCoord[0].xy);

    if (colorMapEnabled) {
        col.rgb = clamp(colorMatrix * col.rgb, 0.0, 1.0);
    }

    gl_FragColor = col;
}

#version 130

uniform sampler2D texture;
uniform vec2 imageSize;
uniform float disperseFactor;
uniform float dimRate;

void main() {
    float xStep = 1.0 / imageSize.x;
    float yStep = 1.0 / imageSize.y;

    vec3 avg = vec3(0.0);
    vec2 myPos = gl_TexCoord[0].xy;

    int count = 0;

    for (int dx = -1; dx <= 1; dx++) {
        float x = myPos.x + xStep * float(dx);

        for (int dy = -1; dy <= 1; dy++) {
            float y = myPos.y + yStep * float(dy);
            vec4 c = texture2D(texture, vec2(x, y));
            avg += c.rgb;
            count++;
        }
    }

    avg /= float(count);

    vec4 original = texture2D(texture, myPos);
    vec3 color = original.rgb + (avg - original.rgb) * disperseFactor - dimRate;

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), original.a);
}

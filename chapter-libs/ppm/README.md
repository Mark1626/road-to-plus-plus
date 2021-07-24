# PPM Library

Ported from this [repo](https://github.com/skeeto/sort-circle), credits to the original author [skeeto](https://github.com/skeeto)

## Usage

Example are found within the examples folder

```cpp
// Create instance
PPM image(800, 600);

// Write the text Hello World
image.write_line("Hello World", 400, 300, colors[1]);

// Write to file
std::ofstream os("test.txt", std::ofstream::out);
image.write(os);
```

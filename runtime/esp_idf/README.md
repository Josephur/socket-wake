# ESP-IDF demo

The v1 demo binary lands in v2. For now, see [DESIGN.md](../../DESIGN.md)
for the plan and [runtime/c/socket_wake.h](../c/socket_wake.h) for the
public surface. The Rust crate compiles with
`cargo build --target xtensa-esp32s3-espidf` (and the equivalent for P4
once that target stabilizes).
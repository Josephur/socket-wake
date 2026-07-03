// SPDX-License-Identifier: Apache-2.0
//! socket-wake: open-source keyword spotting for resource-constrained MCUs.
//! See DESIGN.md for the architecture.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

#[cfg(test)]
mod tests {
    #[test]
    fn scaffold_compiles() {
        assert_eq!(2 + 2, 4);
    }
}
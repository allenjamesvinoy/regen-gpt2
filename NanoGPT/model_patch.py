"""
model_patch.py - How to modify nanoGPT's model.py for custom attention masks

This file documents the EXACT changes needed in nanoGPT/model.py.
The only modification is to CausalSelfAttention.forward() so it accepts
an optional custom_mask parameter. Everything else stays the same.

You can either apply these changes manually or run this script to patch
model.py automatically:

    python model_patch.py

"""

import re
import sys
import os


def patch_model():
    model_path = "model.py"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run this from the nanoGPT root directory.")
        sys.exit(1)

    with open(model_path, "r") as f:
        code = f.read()

    # Back up original
    with open("model_original.py", "w") as f:
        f.write(code)
    print("Backed up original model.py -> model_original.py")

    # =========================================================================
    # PATCH 1: CausalSelfAttention.forward() signature
    # Change:  def forward(self, x):
    # To:      def forward(self, x, custom_mask=None):
    # =========================================================================
    code = code.replace(
        "    def forward(self, x):\n        B, T, C = x.size()",
        "    def forward(self, x, custom_mask=None):\n        B, T, C = x.size()"
    )

    # =========================================================================
    # PATCH 2: Flash attention path - add custom_mask routing
    #
    # nanoGPT uses Flash Attention when available. Flash attention doesn't
    # support arbitrary custom masks easily. So when custom_mask is provided,
    # we fall through to the manual attention path.
    #
    # Find the flash attention block and wrap it in a condition.
    # =========================================================================

    # The flash attention block in nanoGPT looks like:
    #   if self.flash:
    #       y = torch.nn.functional.scaled_dot_product_attention(...)
    #   else:
    #       att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    #       att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    #       ...

    # Replace the flash attention condition to also check for custom_mask
    code = code.replace(
        "        if self.flash:\n",
        "        if self.flash and custom_mask is None:\n"
    )

    # =========================================================================
    # PATCH 3: Manual attention path - apply custom_mask when provided
    #
    # In the else branch, after the standard causal mask is applied,
    # we also apply the custom_mask if provided.
    # =========================================================================

    # Find the line that applies the causal mask in the manual path:
    #   att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    # Add custom_mask application after it.

    old_mask_line = "            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))"
    new_mask_block = """            if custom_mask is not None:
                att = att.masked_fill(custom_mask == 0, float('-inf'))
            else:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))"""

    code = code.replace(old_mask_line, new_mask_block)

    # =========================================================================
    # PATCH 4: Block.forward() - pass custom_mask through
    # Change:  def forward(self, x):
    #              x = x + self.attn(self.ln_1(x))
    # To:      def forward(self, x, custom_mask=None):
    #              x = x + self.attn(self.ln_1(x), custom_mask=custom_mask)
    # =========================================================================

    # Block's forward
    code = code.replace(
        "    def forward(self, x):\n        x = x + self.attn(self.ln_1(x))",
        "    def forward(self, x, custom_mask=None):\n        x = x + self.attn(self.ln_1(x), custom_mask=custom_mask)"
    )

    # =========================================================================
    # PATCH 5: GPT.forward() - accept and pass custom_mask
    # Change:  def forward(self, idx, targets=None):
    # To:      def forward(self, idx, targets=None, custom_mask=None):
    # And change: for block in self.transformer.h:
    #                 x = block(x)
    # To:         for block in self.transformer.h:
    #                 x = block(x, custom_mask=custom_mask)
    # =========================================================================

    code = code.replace(
        "    def forward(self, idx, targets=None):",
        "    def forward(self, idx, targets=None, custom_mask=None):"
    )

    code = code.replace(
        "        for block in self.transformer.h:\n            x = block(x)",
        "        for block in self.transformer.h:\n            x = block(x, custom_mask=custom_mask)"
    )

    with open(model_path, "w") as f:
        f.write(code)

    print("Successfully patched model.py with custom_mask support!")
    print("\nChanges made:")
    print("  1. CausalSelfAttention.forward() accepts custom_mask parameter")
    print("  2. Flash attention bypassed when custom_mask is provided")
    print("  3. Custom mask replaces causal mask when provided")
    print("  4. Block.forward() passes custom_mask through")
    print("  5. GPT.forward() accepts and propagates custom_mask")
    print("\nOriginal saved as model_original.py")


if __name__ == "__main__":
    patch_model()

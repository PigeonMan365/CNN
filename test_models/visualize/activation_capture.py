import torch
from collections import OrderedDict


def capture_activations(model, input_tensor):
    """
    Run a TorchScript model on an input tensor and capture:
      - Activations for each layer
      - Output shapes
      - Execution order of layers

    Returns:
      activations: dict[layer_name] = activation tensor (CPU numpy)
      layer_shapes: dict[layer_name] = tuple shape
      layer_order: list of layer names in execution order
    """

    activations = OrderedDict()
    layer_shapes = OrderedDict()
    layer_order = []

    # ------------------------------------------------------------
    # Hook function
    # ------------------------------------------------------------
    def make_hook(name):
        def hook(module, inp, out):
            # Record execution order
            layer_order.append(name)

            # Save activation (CPU numpy for Streamlit)
            try:
                act = out.detach().cpu().numpy()
            except Exception:
                # Some layers output tuples (e.g., LSTM) — skip them
                return

            activations[name] = act
            layer_shapes[name] = tuple(act.shape)

        return hook

    # ------------------------------------------------------------
    # Register hooks on all submodules
    # ------------------------------------------------------------
    for name, module in model.named_modules():
        # Skip the top-level module
        if name == "":
            continue

        # Attach hook to all modules that produce activations
        module.register_forward_hook(make_hook(name))

    # ------------------------------------------------------------
    # Run the model once to trigger hooks
    # ------------------------------------------------------------
    with torch.no_grad():
        _ = model(input_tensor)

    return activations, layer_shapes, layer_order

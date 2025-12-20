# run_caelonyx.py
import numpy as np
from project_chimera.cognitive_models.caelonyx_model import CaelonyxModel
from project_chimera.l1_calculus.tensor import Tensor
from project_chimera.l1_calculus.optimizers import Adam
from project_chimera.nn.losses import CrossEntropyLoss
import os

# os.environ['PROMETHEUS_USE_GPU'] = 'true' # GPU उपयोग के लिए इसे अनकम्मेंट करें

def main():
    print("===== प्रोजेक्ट Caelonyx: ट्रांसफॉर्मर आर्किटेक्चर डेमो (उन्नत) =====")

    # --- 1. मॉडल हाइपरपैरामीटर्स ---
    vocab_size = 10000
    embed_size = 128
    num_layers = 4
    heads = 4
    forward_expansion = 4
    max_length = 256
    
    # --- 2. Caelonyx मॉडल को इनिशियलाइज़ करें ---
    print("\n[STEP 1] CaelonyxModel को इनिशियलाइज़ किया जा रहा है...")
    model = CaelonyxModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        forward_expansion=forward_expansion,
        max_length=max_length
    )
    print("मॉडल सफलतापूर्वक बन गया!")
    print(f"कुल सीखने योग्य पैरामीटर्स की संख्या: {len(model.parameters())}")

    # --- 3. यथार्थवादी डेटा बनाएं ---
    batch_size = 8
    seq_len = 64
    dummy_input = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    dummy_target = np.random.randint(0, vocab_size, (batch_size * seq_len,))
    print(f"\n[STEP 2] डमी डेटा बनाया गया: input shape={dummy_input.shape}")

    # --- 4. फॉरवर्ड पास चलाएं ---
    print("\n[STEP 3] फॉरवर्ड पास चलाया जा रहा है...")
    logits = model.forward(dummy_input)
    print(f"आउटपुट लॉजिट्स का शेप: {logits.shape}")

    # --- 5. लॉस, बैकवर्ड पास, और ऑप्टिमाइज़र स्टेप ---
    print("\n[STEP 4] लॉस, बैकवर्ड पास, और ऑप्टिमाइज़र स्टेप...")
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    # ट्रेनिंग लूप का एक स्टेप चलाएं
    optimizer.zero_grad()
    
    logits_reshaped = logits.reshape(batch_size * seq_len, vocab_size)
    loss = loss_fn.forward(logits_reshaped, dummy_target)
    print(f"प्रारंभिक लॉस: {loss.data.item():.4f}")

    loss.backward()
    print("बैकवर्ड पास पूरा हुआ। ग्रेडिएंट्स की गणना हो गई है।")
    
    # कुछ ग्रेडिएंट्स की जाँच करें
    first_param = model.parameters()[0]
    if first_param.grad is not None:
        grad_norm = np.linalg.norm(first_param.grad)
        print(f"पहले पैरामीटर के ग्रेडिएंट का नॉर्म: {grad_norm}")
        if grad_norm == 0:
            print("चेतावनी: ग्रेडिएंट 0 है! बैकप्रॉप में कोई समस्या हो सकती है।")

    optimizer.step()
    print("ऑप्टिमाइज़र स्टेप पूरा हुआ। मॉडल के वेट्स अपडेट हो गए हैं।")

    # एक और फॉरवर्ड पास यह देखने के लिए कि लॉस बदला है या नहीं
    logits_after_step = model.forward(dummy_input)
    logits_reshaped_after = logits_after_step.reshape(batch_size * seq_len, vocab_size)
    loss_after_step = loss_fn.forward(logits_reshaped_after, dummy_target)
    print(f"एक स्टेप के बाद लॉस: {loss_after_step.data.item():.4f}")

    print("\n===== डेमो सफलतापूर्वक पूरा हुआ! =====")

if __name__ == "__main__":
    main()

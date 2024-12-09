import numpy as np


def softmax(logits):  # end
    exp_logits = np.exp(logits)  # חישוב האקספוננט של כל הלוגיטים
    return exp_logits / np.sum(exp_logits)  # חישוב ההסתברויות

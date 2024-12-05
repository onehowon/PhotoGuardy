from art.attacks.evasion import (
    FastGradientMethod, CarliniL2Method, DeepFool, AutoAttack,
    ProjectedGradientDescent, BasicIterativeMethod, SpatialTransformation,
    MomentumIterativeMethod, SaliencyMapMethod, NewtonFool
)

def generate_adversarial_image(img_tensor, classifier, attack_type, eps_value):
    if attack_type == "FGSM":
        attack = FastGradientMethod(estimator=classifier, eps=eps_value)
    elif attack_type == "C&W":
        attack = CarliniL2Method(classifier=classifier, confidence=0.05)
    elif attack_type == "DeepFool":
        attack = DeepFool(classifier=classifier, max_iter=20)
    elif attack_type == "AutoAttack":
        attack = AutoAttack(estimator=classifier, eps=eps_value, batch_size=1)
    elif attack_type == "PGD":
        attack = ProjectedGradientDescent(estimator=classifier, eps=eps_value, eps_step=eps_value / 10, max_iter=40)
    elif attack_type == "BIM":
        attack = BasicIterativeMethod(estimator=classifier, eps=eps_value, eps_step=eps_value / 10, max_iter=10)
    elif attack_type == "STA":
        attack = SpatialTransformation(estimator=classifier, max_translation=0.2)
    elif attack_type == "MIM":
        attack = MomentumIterativeMethod(estimator=classifier, eps=eps_value, eps_step=eps_value / 10, max_iter=10)
    elif attack_type == "JSMA":
        attack = SaliencyMapMethod(classifier=classifier, theta=0.1, gamma=0.1)
    elif attack_type == "NewtonFool":
        attack = NewtonFool(classifier=classifier, max_iter=20)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")
    
    adv_img_np = attack.generate(x=img_tensor.cpu().numpy())
    return adv_img_np

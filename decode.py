import os
import torch
import numpy as np
import utils
import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_0bit(imgs, carrier, angle, model):
    """
    0-bit watermarking detection.

    Args:
        imgs: List of PIL images
        carrier: Hypercone direction 1xD
        angle: Angle of the hypercone
        model: Neural net model to extract the features

    Returns:
        List of decoded datum as a dictionary for each image.
        Contains the following fields:
            - R: acceptance function of the hypercone, positive when x lies in the cone, negative otherwise
            - log10_pvalue: log10 of the p-value, i.e. if we were drawing O(1/pvalue) random carriers, 
                on expectation, one of them would give an R bigger or equal to the one that is observed.
    """
    rho = 1 + np.tan(angle)**2
    decoded_data = []

    for ii, img in enumerate(imgs):
        img = utils_img.default_transform(img).unsqueeze(0).to(device, non_blocking=True) # 1xCxHxW
        ft = model(img) # 1xCxWxH -> 1xD
        dot_product = (ft @ carrier.T).squeeze() # 1xD @ Dx1 -> 1
        norm = torch.norm(ft, dim=-1) # 1xD -> 1 
        R = (rho * dot_product**2 - norm**2).item()
        cosine = torch.abs(dot_product/norm)
        log10_pvalue = np.log10(utils.cosine_pvalue(cosine.item(), ft.shape[-1]))
        decoded_data.append({'index': ii, 'R': R, 'log10_pvalue': log10_pvalue})
    
    return decoded_data


def decode_trace_carriers(imgs, model, angle, carriers_dir, threshold):
    """
    Trace watermarking: compare each image against every carrier in carriers_dir.
    
    Args:
        imgs: List of PIL images
        model: Neural net model to extract the features
        angle: Angle of the hypercone
        carriers_dir: Directory containing .pth carrier files
        threshold: Confidence threshold for matching (e.g. 0.95)
    
    Returns:
        List of dicts with keys:
            index, carrier_name, R, log10_pvalue, confidence, matched
    """
    rho = 1 + np.tan(angle)**2
    decoded_data = []
    
    carrier_files = sorted([
        f for f in os.listdir(carriers_dir) if f.endswith('.pth')
    ])
    
    for carrier_file in carrier_files:
        carrier_path = os.path.join(carriers_dir, carrier_file)
        carrier = torch.load(carrier_path, map_location=device)
        carrier = carrier.to(device, non_blocking=True)
        carrier_name = os.path.splitext(carrier_file)[0]
        
        for ii, img in enumerate(imgs):
            try:
                img_tensor = utils_img.default_transform(img).unsqueeze(0).to(device, non_blocking=True)
                ft = model(img_tensor)
                dot_product = (ft @ carrier.T).squeeze()
                norm = torch.norm(ft, dim=-1)
                R = (rho * dot_product**2 - norm**2).item()
                cosine = torch.abs(dot_product / norm)
                log10_pvalue = np.log10(utils.cosine_pvalue(cosine.item(), ft.shape[-1]))
                confidence = 1 - 10**log10_pvalue
                matched = confidence > threshold
            except (RuntimeError, ValueError):
                # multibit carriers (K>1) or shape mismatch: cannot reduce to scalar
                R = None
                log10_pvalue = None
                confidence = None
                matched = False
            decoded_data.append({
                'index': ii,
                'carrier_name': carrier_name,
                'R': R,
                'log10_pvalue': log10_pvalue,
                'confidence': confidence,
                'matched': matched,
            })
    
    return decoded_data


def decode_multibit(imgs, carrier, model):
    """
    multi-bit watermarking decoding.

    Args:
        imgs: List of PIL images
        carrier (tensor of size KxD): K carriers of dimension D, each one determines a bit
        model: Neural net model to extract the features

    Returns:
        List of decoded datum as a dictionary for each image.
        Contains the following fields:
            - msg: message extracted from the watermark as a tensor of booleans
    """
    decoded_data = []
    for ii, img in enumerate(imgs):
        img = utils_img.default_transform(img).unsqueeze(0).to(device, non_blocking=True) # 1xCxHxW
        ft = model(img) # 1xCxWxH -> 1xD
        dot_product = ft @ carrier.T # 1xD @ DxK -> 1xK
        msg = torch.sign(dot_product).squeeze() > 0
        msg = msg.cpu()
        decoded_data.append({'index': ii, 'msg': msg})

    return decoded_data

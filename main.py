import cv2
import numpy as np

def rotate(image, angle):
    h, w = image.shape[:2]
    # 画像をRGBA形式に変換（透明背景用）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # 回転後のサイズを計算
    radian = np.radians(angle)
    sine = np.abs(np.sin(radian))
    cosine = np.abs(np.cos(radian))
    tri_mat = np.array([[cosine, sine], [sine, cosine]], np.float32)
    old_size = np.array([w, h], np.float32)
    new_size = np.ravel(np.dot(tri_mat, old_size.reshape(-1, 1)))
    
    # 回転アフィン行列を生成
    affine = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    affine[:2, 2] += (new_size - old_size) / 2.0
    affine[:2, :] *= (old_size / new_size).reshape(-1, 1)
    
    rotated_image = cv2.warpAffine(image, affine, (w, h))
    
    # 高さを半分にリサイズ
    half_height = h // 2
    resized_image = cv2.resize(rotated_image, (w, half_height))

    return resized_image

if __name__ == "__main__":
    ruleSet = [
        {"input": "input/none.png", "output": "output/-.png", "rotate": 45},
        {"input": "input/straight.png", "output": "output/ns.png", "rotate": 45},
        {"input": "input/straight.png", "output": "output/ew.png", "rotate": -45},
        {"input": "input/end.png", "output": "output/e.png", "rotate": 45},
        {"input": "input/end.png", "output": "output/s.png", "rotate": -45},
        {"input": "input/end.png", "output": "output/n.png", "rotate": 135},
        {"input": "input/end.png", "output": "output/w.png", "rotate": -135},
        {"input": "input/corner.png", "output": "output/ne.png", "rotate": 45},
        {"input": "input/corner.png", "output": "output/se.png", "rotate": -45},
        {"input": "input/corner.png", "output": "output/nw.png", "rotate": 135},
        {"input": "input/corner.png", "output": "output/sw.png", "rotate": -135},
        {"input": "input/t_junction.png", "output": "output/new.png", "rotate": 45},
        {"input": "input/t_junction.png", "output": "output/nse.png", "rotate": -45},
        {"input": "input/t_junction.png", "output": "output/nsw.png", "rotate": 135},
        {"input": "input/t_junction.png", "output": "output/sew.png", "rotate": -135},
        {"input": "input/diagonal.png", "output": "output/d_ne.png", "rotate": 45},
        {"input": "input/diagonal.png", "output": "output/d_se.png", "rotate": -45},
        {"input": "input/diagonal.png", "output": "output/d_nw.png", "rotate": 135},
        {"input": "input/diagonal.png", "output": "output/d_sw.png", "rotate": -135},
        {"input": "input/cross.png", "output": "output/nsew.png", "rotate": 45},
    ]

    for rule in ruleSet:
        print(rule)
        image = cv2.imread(rule["input"])
        image = rotate(image, rule["rotate"])
        cv2.imwrite(rule["output"], image)

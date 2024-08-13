import cv2
import numpy as np
import json
import argparse

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

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"JSONファイルの読み込み中にエラーが発生しました: {file_path}")


if __name__ == "__main__":
    # コマンドライン引数をパースする
    parser = argparse.ArgumentParser(description="指定されたJSONファイルを読み込むスクリプト")
    parser.add_argument("file_name", help="読み込みたいJSONファイルのパス")
    args = parser.parse_args()

    # ファイル名を引数から取得
    file_name = args.file_name
    try:
        json_data = load_json(file_name)
        for rule in json_data:
            print(rule)
            image = cv2.imread(rule["input"])
            image = rotate(image, rule["rotate"])
            cv2.imwrite(rule["output"], image)

    except Exception as e:
        print(f"エラーが発生しました: {e}")


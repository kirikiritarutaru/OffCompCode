import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


if __name__ == '__main__':
    # テストデータの生成
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_proba = np.clip(y_true + np.random.normal(0, 0.3, 100), 0, 1)

    result = threshold_search(y_true, y_proba)

    print("最適な閾値:", result['threshold'])
    print("F1スコア:", result['f1'])

    # 必要に応じて、結果の妥当性をアサート文でチェック
    assert 0 <= result['threshold'] <= 1
    assert 0 <= result['f1'] <= 1

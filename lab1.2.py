import numpy as np

# Функція активації (ступенева функція Хевісайда)
def step_function(x):
    return 1 if x >= 0 else 0

# Логічні дані для навчання
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

# Очікувані результати
expected_outputs = np.array([1, 1, 0, 1])

# Навчання нейронної мережі
def train(weights, bias):
    for input_data, expected_output in zip(inputs, expected_outputs):
        # Сума взважених вхідних сигналів та зсув
        weighted_sum = np.dot(input_data, weights) + bias
        # Вихід нейрону
        output = step_function(weighted_sum)
        # Оновлення ваг та зсуву
        error = expected_output - output
        weights += error * input_data
        bias += error
    return weights, bias

# Тестування нейронної мережі
def test(weights, bias):
    for input_data, expected_output in zip(inputs, expected_outputs):
        # Сума взважених вхідних сигналів та зсув
        weighted_sum = np.dot(input_data, weights) + bias
        # Вихід нейрону
        output = step_function(weighted_sum)
        print(f"Input: {input_data}, Output: {output}, Expected: {expected_output}")

# Навчання та тестування нейронної мережі
weights = np.array([0.5, 0.5])  # ваги
bias = -0.5  # зсув
weights, bias = train(weights, bias)
print("Trained Weights:", weights)
print("Trained Bias:", bias)
print("\nTesting:")
test(weights, bias)

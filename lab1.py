import numpy as np
import matplotlib.pyplot as plt

# Логічна функція І (AND)
def logical_and():
    # Визначаємо вхідні дані та очікувані результати
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])

    # Ініціалізуємо ваги та зсув
    weights = np.random.rand(2, 1)
    bias = np.random.rand(1)

    # Налаштовуємо гіперпараметри
    lr = 0.1
    epochs = 1000

    # Навчання моделі
    for epoch in range(epochs):
        inputs = X
        
        # Подання на входи та обчислення вагованих сум
        weighted_sum = np.dot(inputs, weights) + bias
        
        # Активаційна функція (порогова)
        activated_output = np.where(weighted_sum >= 0, 1, 0)
        
        # Обчислення помилки та корекція вагів
        error = y - activated_output
        weights += lr * np.dot(inputs.T, error)
        bias += lr * np.sum(error)

    # Виведення результатів
    print("Результати після навчання:")
    print("Ваги: ")
    print(weights)
    print("Зсув: ")
    print(bias)

    # Перевірка на тестових даних
    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    result = np.dot(test_input, weights) + bias
    print("Результати передбачення: ")
    print(np.where(result >= 0, 1, 0))

# Логічна функція АБО (OR)
def logical_or():
        # Визначаємо вхідні дані та очікувані результати
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])

    # Ініціалізуємо ваги та зсув
    weights = np.random.rand(2, 1)
    bias = np.random.rand(1)

    # Налаштовуємо гіперпараметри
    lr = 0.1
    epochs = 1000

    # Навчання моделі
    for epoch in range(epochs):
        inputs = X
        
        # Подання на входи та обчислення вагованих сум
        weighted_sum = np.dot(inputs, weights) + bias
        
        # Активаційна функція (порогова)
        activated_output = np.where(weighted_sum >= 0, 1, 0)
        
        # Обчислення помилки та корекція вагів
        error = y - activated_output
        weights += lr * np.dot(inputs.T, error)
        bias += lr * np.sum(error)

    # Виведення результатів
    print("Результати після навчання:")
    print("Ваги: ")
    print(weights)
    print("Зсув: ")
    print(bias)

    # Перевірка на тестових даних
    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    result = np.dot(test_input, weights) + bias
    print("Результати передбачення: ")
    print(np.where(result >= 0, 1, 0))

# Логічна функція НІ (NOT)
def logical_not():
        # Визначаємо вхідні дані та очікувані результати
    X = np.array([[0], [1]])
    y = np.array([[1], [0]])

    # Ініціалізуємо ваги та зсув
    weights = np.random.rand(1, 1)
    bias = np.random.rand(1)

    # Налаштовуємо гіперпараметри
    lr = 0.1
    epochs = 1000

    # Навчання моделі
    for epoch in range(epochs):
        inputs = X
        
        # Подання на входи та обчислення вагованих сум
        weighted_sum = np.dot(inputs, weights) + bias
        
        # Активаційна функція (порогова)
        activated_output = np.where(weighted_sum >= 0, 1, 0)
        
        # Обчислення помилки та корекція вагів
        error = y - activated_output
        weights += lr * np.dot(inputs.T, error)
        bias += lr * np.sum(error)

    # Виведення результатів
    print("Результати після навчання:")
    print("Ваги: ")
    print(weights)
    print("Зсув: ")
    print(bias)

    # Перевірка на тестових даних
    test_input = np.array([[0], [1]])
    result = np.dot(test_input, weights) + bias
    print("Результати передбачення: ")
    print(np.where(result >= 0, 1, 0))

# Логічна функція Виключне АБО (XOR)
def logical_xor():
        # Визначаємо вхідні дані та очікувані результати
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Ініціалізуємо ваги та зсув
    weights_input_hidden = np.random.rand(2, 2)
    bias_input_hidden = np.random.rand(1, 2)

    weights_hidden_output = np.random.rand(2, 1)
    bias_hidden_output = np.random.rand(1)

    # Налаштовуємо гіперпараметри
    lr = 0.1
    epochs = 10000

    # Навчання моделі
    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_input_hidden
        hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
        
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_hidden_output
        activated_output = 1 / (1 + np.exp(-output_layer_input))
        
        # Backpropagation
        error = y - activated_output
        d_predicted_output = error * (activated_output * (1 - activated_output))
        
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * (hidden_layer_output * (1 - hidden_layer_output))
        
        # Updating Weights and Biases
        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * lr
        bias_hidden_output += np.sum(d_predicted_output) * lr
        
        weights_input_hidden += X.T.dot(d_hidden_layer) * lr
        bias_input_hidden += np.sum(d_hidden_layer) * lr

    # Виведення результатів
    print("Результати після навчання:")
    print("Ваги (вхідний -> прихований шар): ")
    print(weights_input_hidden)
    print("Зсув (вхідний -> прихований шар): ")
    print(bias_input_hidden)
    print("Ваги (прихований -> вихідний шар): ")
    print(weights_hidden_output)
    print("Зсув (прихований -> вихідний шар): ")
    print(bias_hidden_output)

    # Перевірка на тестових даних
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_input_hidden
    hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_hidden_output
    activated_output = 1 / (1 + np.exp(-output_layer_input))
    print("Результати передбачення: ")
    print(np.round(activated_output))

# Прогнозування часового ряду
def time_series_prediction():
        # Вихідні дані (часовий ряд)
    data = np.array([2.5, 4.2, 1.6, 4.2, 1.1, 4.4, 0.8, 4.1, 0.0, 4.7, 1.9, 4.1, 0.0, 5.0, 1.4])

    # Параметри моделі
    window_size = 3  # Розмір вікна
    lr = 0.001       # Швидкість навчання
    epochs = 1000    # Кількість епох

    # Створення даних для навчання
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    X = np.array(X)
    y = np.array(y)

    # Ініціалізація вагів та зсуву
    np.random.seed(0)
    weights = np.random.rand(window_size)
    bias = np.random.rand(1)

    # Навчання моделі
    for epoch in range(epochs):
        for i in range(len(X)):
            prediction = np.dot(X[i], weights) + bias
            error = y[i] - prediction
            weights += lr * error * X[i]
            bias += lr * error

    # Передбачення на основі навченої моделі
    predictions = []
    for i in range(len(X)):
        prediction = np.dot(X[i], weights) + bias
        predictions.append(prediction[0])

    # Виведення результатів та графіка
    plt.plot(data, label='Дійсні значення')
    plt.plot(np.arange(window_size, len(data)), predictions, label='Прогнозовані значення')
    plt.xlabel('Час')
    plt.ylabel('Значення')
    plt.legend()
    plt.show()

# Виклик функцій
logical_and()
logical_or()
logical_not()
logical_xor()
time_series_prediction()

plt.show()  # Показати графіки (якщо потрібно)

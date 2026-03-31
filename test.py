import cv2
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import cv2

def figure_to_array(fig):
    """Преобразует matplotlib figure в numpy массив для OpenCV."""
    # 1. Отрисовываем фигуру
    fig.canvas.draw()
    
    # 2. Получаем RGBA буфер (это стандарт для новых версий Matplotlib)
    # Используем buffer_rgba() вместо устаревшего tostring_rgb()
    rgba_buffer = fig.canvas.buffer_rgba()
    
    # 3. Превращаем в numpy массив
    img_array = np.array(rgba_buffer)
    
    # 4. Конвертируем из RGBA (Matplotlib) в BGR (OpenCV)
    # Обратите внимание: OpenCV ожидает BGR, а у нас 4 канала (RGBA)
    return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

# --- Пример использования ---
fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
ax.plot([1, 2, 3], [10, 20, 10], 'r-o')
ax.set_title("График из модели")

# Конвертируем фигуру в картинку
plot_img = figure_to_array(fig)

# Показываем через OpenCV
cv2.imshow('Matplotlib in OpenCV', plot_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
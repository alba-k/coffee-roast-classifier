# Datos propios del proyecto — Coffee Roast Classifier

Estado al 16 de octubre de 2026. Esta ficha describe lo que se hizo y se midió en el proyecto, para usarla como fuente en el informe junto con las referencias. Los números son los medidos en el proyecto; lo marcado como "pendiente" aún no existe.

## Objetivo
Clasificar el nivel de tostado de granos de café a partir de una imagen (una imagen = una etiqueta) con Google Teachable Machine y, después, ejecutar una acción física con un kit ESP32 (anillo de LED o servo) y Python.

## Montaje de captura
- Caja cerrada por los cuatro lados, con una cámara web USB (1280x720) y un anillo de 16 LED controlado por un ESP32-S3.
- Luz ajustada para un blanco neutro-cálido (mezcla R 230, G 235, B 100, brillo 230), después de que una luz fría daba un tinte azulado.
- Una foto cada 3 segundos, guardada en JPEG (calidad 92) de hasta 800 px de ancho, con tres formatos de colocación: grupo, disperso e individual.

## Clases
Verde, Claro, Medio, MedioOscuro, Oscuro, Quemado y Fondo (nada_otro: papel vacío, manos y otros objetos). Se descartó una clase "defectuoso".

## Dataset
- Fotos grabadas: 2.592 (verde 352, claro 370, medio 377, medio_oscuro 372, oscuro 386, quemado 380, nada_otro 355).
- Filtrado automático con reglas propias: manos y guantes, pocas semillas visibles, desenfoque (no aplicado a granos individuales), exposición y mezcla de tostados en la clase medio. Quedaron 2.344 fotos.
- Recorte al centro de 450x450 px para quitar el borde oscuro de la cámara; quedaron 2.328 fotos.
- División por bloques de 5 fotos consecutivas (para que fotos casi idénticas no queden en entrenamiento y prueba a la vez): entrenamiento 1.628, validación 470, prueba 230.
- Solo se subió a Teachable Machine la carpeta de entrenamiento; validación y prueba se guardaron para medir el modelo con fotos que nunca vio.

## Modelo
Teachable Machine (modelo estándar de imágenes, 224x224 en color, basado en una red tipo MobileNet) exportado como Keras (.h5) y SavedModel, y convertido a ONNX para ejecutarlo con OpenCV sin instalar TensorFlow. La conversión reproduce la salida original con una diferencia de ~2e-7.

## Resultados
- Con fotos guardadas: 99,6% en validación (468 de 470) y 99,6% en prueba (229 de 230).
- Con brillo ±15% y tintes cálidos o fríos aplicados a esas fotos, el modelo siguió acertando todas las muestras probadas (28 por clase).
- Con la cámara en vivo, las clases Medio y Quemado fallaron (por ejemplo, Quemado se clasificó como MedioOscuro).

## Diagnóstico: sesgo de fondo (atajo)
Se midió la "suciedad fina" del papel (variación de textura en el 40% de píxeles más claros) en las fotos de entrenamiento: verde 5,0; claro 3,9; medio 5,6; medio_oscuro 4,7; oscuro 10,8; quemado 12,2. Oscuro y Quemado se grabaron sobre papel más sucio que las demás clases, y el modelo pudo aprender el fondo como pista. Es el fenómeno descrito en la literatura como shortcut learning (Geirhos et al., 2020) y sesgo de fondo (Xiao et al., 2020).

## Corrección en curso
Regrabar todas las clases sobre una hoja blanca limpia (segundo fondo), conservando las fotos anteriores, de modo que cada clase tenga fotos de los dos fondos: Quemado, Oscuro y Medio 150 fotos cada una; MedioOscuro, Claro, Verde y Fondo 100 cada una. Luego reentrenar y volver a medir en vivo. Resultado: pendiente.

## Lecciones
1. La exactitud que muestra Teachable Machine mezcla fotos parecidas y es optimista; hay que medir con fotos separadas.
2. Una exactitud muy alta con fotos guardadas no garantiza buen funcionamiento en vivo.
3. Cada clase debe compartir las mismas condiciones de fondo y luz para que el modelo aprenda el grano y no el entorno.

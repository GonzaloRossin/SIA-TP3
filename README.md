# SIA-TP3

## Librerias Necesarias

Para poder ejecutar el motor es necesario poseer las siguientes librerias de python:

- matplotlib
- numpy

En caso de no tenerlas instaladas, se pueden ejecutar los siguientes comandos para hacerlo:

`pip install matplotlib`

`pip install numpy`

## Ejecución del proyecto

### Ejercicio 1

#### Item a:

##### Ejecutar programa

Debe estar en carpeta EJ1/a:

```bash
    python plotter.py
```

#### Administrar parámetros

Para administrar parámetros hay un archivo EJ1config.json,donde podemos gestionar varios parámetros diferentes de nuestro programa como la cantidad de capas ocultas, el numero de epocas para entrenamiento y los parametros de los optimizadores.

```json
{
    "operation": "and",
    "learning_rate": 0.001,
    "num_epochs": 100,
    "hidden_layers":[17,2,17],
    
    "optimizer": {
        "method": "adam",
        "momentum_alpha": 0.9,
        "adam": {
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        }
    },
    
    "num_epochs": 2000,
    "num_features": 35,
    "num_outputs": 35,
}
```

### Ejercicio 2

## Librerias Necesarias

- tensorflow
- scipy

En caso de no tenerlas instaladas, se pueden ejecutar los siguientes comandos para hacerlo:

`pip install tensorflow`

`pip install scipy`

#### Ejecución del proyecto

Para ejecutar el ejercicio 2, uno debe moverse al directorio llamado "EJ2" por medio del siguiente comando mientras se esta ubicado en la carpeta raíz del proyecto:

```bash
    cd EJ2
```

Para correr el ejercicio se utiliza el siguiente comando ubicado dentro del directorio para correr el ejercicio:

```bash
    python main.py
```
es posible cambiar la cantidad de epocas que se va a entrenar la red neuronal asi como ajustar el ancho y el alto de las imagenes que se van a utilizar para entrenar la red. Las imagenes utilizadas para el entrenamiento estan alojadas en la carpeta `images` ubicada dentro del directorio es posible cambiar la cantidad de epocas que se va a entrenar la red neuronal asi como ajustar el ancho y el alto de las imagenes que se van a utilizar para entrenar la red. Las imagenes utilizadas para el entrenamiento estan alojadas en la carpeta `images` ubicada dentro del directorio `EJ2`

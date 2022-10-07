### Folder to test the load_datasets methods

En esta carpeta de test se llevarán a cabo los tests de los métodos de carga de datasets de TensorFlow, PyTorch o HuggingFace. Se ha separado
de la rama principal de tests debido al tiepo que lleva la ejecución de estos tests. 

Para comprobar que los tests funcionan correctamente se debe ejecutar el siguiente comando:

pytest --cov-report term-missing --cov=flex tests_datasets/

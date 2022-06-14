### Comandos de Testeo

Para llevar comprobar que los tests cubren el 100% de las líneas escritas, se ha utilizado el siguiente comando:

pytest --cov-report term-missing --cov=flex tests/

El parámetro *term-missing* indica que debe mostrar el reporte por terminal. Se pueden utilizar las siguientes opciones: ['term', 'term-missing', 'annotate', 'html', 'xml']

Esto devuelve un report de cada archivo mostrando lo siguiente:

- **Stmts**: *statements* que hay que comprobar.
- **Miss**: Cuantos *statements* quedan por comprobar.
- **Cover**: El % de *statements* cubiertos por los tests.
- **Missing**: Líneas de código que quedan por cubrir. En caso de que **Cover** sea 100%, entonces no mostrará nada.
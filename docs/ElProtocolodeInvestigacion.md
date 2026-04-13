<div style="text-align: justify;">

**Titulo:** *Métodos de IA para la homologacion, integracion y analisis de bases de datos de salud y seguridad*

**Clave:** CBF-2025-G-1031

**Responsable Técnico/a:** Dr. Víctor Míreles Chávez

**Convocatoria:** Ciencia Básica y de Frontera 2025. Protocolo de investigación

---

# Métodos de IA para la homologacion, integracion y analisis de bases de datos de salud y seguridad

| Participantes                        | Institución                                                                 |
|--------------------------------------|-----------------------------------------------------------------------------|
| Dr. Víctor Míreles Chávez (CVU 41396) | Centro de Investigaciones Interdisciplinarias en Ciencias y Humanidades de la Universidad Nacional Autónoma de México (RENIECYT 1602701-30) |
| Dra. Mariana Esther Martínez Sánchez (CVU 400554) | Instituto Nacional de Enfermedades Respiratorias Ismael Cosío Villegas (RENIECYT 1800577) |
| Dr. Víctor Hugo Muñiz Sánchez (CVU 41539) | Centro de Investigación en Matemáticas A.C. Unidad Monterrey (RENIECYT 1800236-2) |

# Protocolo de Investigación

## Resumen
El presente proyecto busca empujar la frontera del conocimiento en inteligencia artificial y ciencia de datos, con el propósito de desarrollar métodos innovadores que optimicen la homologación, integración, recombinación, deduplicación y filtrado de datos.

La transformación digital en el sector público ha generado retos significativos en términos de interoperabilidad, calidad y confiabilidad de los datos. Problemas como la duplicidad de registros, la ausencia de estandarización y la dificultad para vincular información dispersa limitan la capacidad de las instituciones de gobierno de tomar decisiones basadas en evidencia. Ante este panorama, el presente estudio busca avanzar en el desarrollo de herramientas computacionales que permitan superar estos obstáculos mediante el uso de técnicas de vanguardia en **procesamiento de lenguaje natural**, **matching difuso** y **modelos grandes de lenguaje**.

La metodología del proyecto se basa en la exploración, diseño e implementación de sistemas que permitan mejorar la precisión y eficiencia en la vinculación de datos heterogéneos. Se abordarán tres ejes fundamentales: **comparación de dos cadenas de texto**, **comparación entre una cadena de texto y catálogo**, y **exploración de estrategias de comparación entre redes de información asociadas a una persona o entidad**.
En todos los casos, los modelos habrán de regresar un score de similitud y una medida de incertidumbre asociada al score, además de contemplar la posibilidad de información faltante y ser escalables para ser aplicados a grandes conjuntos de datos. El proyecto se centrará en el análisis de tres conjuntos de datos específicos: **registros de la Fiscalía General de Justicia de la Ciudad de México**, **información clínica del Instituto Nacional de Enfermedades Respiratorias** y **expedientes del archivo histórico del Servicio Médico Forense**.

Para el desarrollo de estas metodologías se hará una revisión continua del estado del arte para determinar los enfoques y modelos existentes. En paralelo, se evaluará la estructura y calidad de los datos de los conjuntos de ejemplo para identificar los problemas más comunes en los datos y definir los criterios de validación. A continuación se diseñarán e implementarán modelos de inteligencia artificial correspondientes, tomando en consideración la naturaleza sensible de los datos. Los métodos desarrollados serán sometidos a pruebas con los datos proporcionados por las instituciones participantes. La validación incluirá medición de precisión, recall y tasas de error, comparación con técnicas existentes e iteración y ajuste de los modelos. Para garantizar la aplicabilidad del proyecto, se consolidarán los modelos desarrollados en una biblioteca de software.

El impacto de este proyecto se proyecta en la mejora sustancial de los procesos de integración y análisis de información en instituciones públicas, coadyuvando a la realización de sus funciones sustantivas y a la toma de decisiones basada en evidencia. La generación de conocimiento en la intersección de la inteligencia artificial y la gestión de datos contribuirá no solo al desarrollo de nuevas herramientas computacionales, sino también al fortalecimiento de capacidades técnicas en diversos sectores estratégicos del país. Además, el proyecto contempla la incorporación y capacitación de estudiantes en el desarrollo e implementación de modelos de inteligencia artificial. Se realizarán actividades de acceso universal al conocimiento dirigidas a la capacitación de personal de instituciones gubernamentales en el uso y aplicación de las herramientas generadas. Se diseñarán materiales educativos, talleres, manuales técnicos y recursos audiovisuales orientados a facilitar la adopción de estas tecnologías en el ámbito institucional, promoviendo así la implementación de prácticas de análisis de datos más eficientes y transparentes.


## Antecedentes

Actualmente, la información almacenada y/o generada por varias instituciones del sector público se encuentra dispersa en diferentes bases de datos y fuentes de información. Lograr una integración de todas las fuentes de datos es indispensable para tener una información completa de cada entidad (por ejemplo **individuos**, **eventos**, **enfermedades** o **lugares**), y los análisis subsecuentes que se realicen de los mismos. Para ello es necesaria la **identificación de registros de una base de datos que se corresponden a registros en otra**, siendo ambas bases de datos posiblemente de distinta naturaleza (e.g. relacionales, de documentos, etc).

**El problema de la identificación de entidades comunes entre bases de datos** se conoce en la literatura con diversos nombres, entre ellos **ligado de registros**, **homologación de bases de datos**, y **resolución de entidades**. Dicho problema se ha abordado desde antes de la popularización de las computadoras digitales, por ejemplo en el caso de registros médicos [20], y los primeros abordajes teóricos son anteriores a la aparición de los sistemas de bases de datos relacionales [21].

Esta tarea ha sido abordada desde hace bastante tiempo para **comparaciones de dos cadenas de texto de longitudes $n$ y $m$**, con algoritmos de complejidad variable que van desde $O(m)$ u $O(n)$ en el mejor caso (*suffix trees*), hasta $O(nm)$ en el peor caso (*naive search*) [1][2], y algunos eficientes para búsquedas más generales con complejidad $O(m+n)$ en promedio [3]. Esto, para búsquedas exactas. Otro enfoque es la **comparación y búsqueda de cadenas de texto similares, pero no necesariamente idénticas**, algo que puede ser útil cuando se tienen palabras o textos incompletos o incorrectos debido a procesos de digitalización (como OCR) o errores de captura, un escenario muy parecido a lo que se realiza en éste proyecto.
Los métodos para búsquedas aproximadas, también llamado *fuzzy string matching*, **utiliza diversas métricas o distancias que permiten comparar cadenas de texto**, y han sido usadas para diversas de procesamiento de lenguaje natural (PLN), como corrección ortográfica, autocompletar palabras, deduplicación de registros, e incluso detección de plagio. Las distancias más comunes incluyen aquellas basadas en operaciones de edición, como **Levenshtein** [4], **Damerau-Levenshtein** [5] o **Jaro-Winkler** [6][7]; medidas de similaridad con $N-$grams [8] o usando la **distancia coseno** con representaciones TF-IDF para cadenas grandes de texto o documentos [9]. **Sin embargo, una de las deficiencias de éstas métricas, es que se basan principalmente en características morfológicas y léxicas de palabras o N-gramas, sin considerar otras características importantes del lenguaje, como la semántica y contextualidad**.


**Un cambio radical surgió con las representaciones de texto basado en word embeddings, las cuáles son representaciones vectoriales densas de tamaño predeterminado que son capaces de capturar relaciones semánticas y sintácticas muy complejas**. Estas representaciones se aprenden con modelos de lenguaje basados principalmente en redes neuronales profundas que se entrenan en cantidades masivas de texto [10] [11] [12].
Modelos de embeddings ahora básicos, como **Word2Vec** [13], **GloVe** [14] o **FastText** [15], ya mostraban capacidades destacadas para identificar sinonimias y resolver analogías complejas mediante simple aritmética vectorial, lo que **mejoró de manera notable diversas tareas de PLN como análisis de sentimiento, traducción automática, búsqueda semántica y en general, la recuperación de información en grandes bases de datos, donde los embeddings se usaban como modelos pre-entrenados para ser las variables de entrada de diversos modelos de aprendizaje**. Las limitaciones de éstos modelos para modelar contextualidad, se vio resuelta con el uso de arquitecturas de redes neuronales para datos secuenciales, como ELMO [16], que obtiene embeddings usando redes bidireccionales LSTM, **pero sobre todo, con las arquitecturas basadas en Transformers, como BERT [17] o GPT [18], que inauguraban la era de los modelos de lenguaje grandes (LLM)**. Estas representaciones contextuales permiten obtener distancias entre palabras o sentencias que identifican la diferencia entre una misma palabra que se usa en contextos diferentes, por ejemplo, la palabra *“banco”* cuando se utiliza en la frase *“tengo una cuenta en el banco”* o cuando se usa en *“me sentaré en el banco”*. Para muchas aplicaciones, incluidas las de ésta propuesta, es de gran utilidad tener representaciones contextualizadas para registros que incluyen texto en lenguaje libre, **como expedientes clínicos o investigaciones judiciales**, donde es importante realizar procesos de desambiguación de palabras.

Otra actividad de relevancia para éste proyecto, es el **análisis de redes de información entre entidades**, por ejemplo, las que se pueden encontrar en un **evento de desaparición de personas donde se pueden identificar víctima, perpetrador, autoridades, fecha y lugar de desaparición como entidades nombradas en un expediente de investigación**. Esto requiere identificar automáticamente las entidades nombradas (NER), que es una tarea de PLN donde se identifican y clasifican entidades nombradas en categorías predefinidas como personas, organizaciones, lugares, fechas, entre otras. Actualmente, esta tarea se aborda con métodos de aprendizaje profundo [19], y principalmente, con LLMs como BERT, que puede entrenarse para NER además de las tareas tradicionales de modelo de lenguaje enmascarado y predicción de la próxima sentencia. Si bien estas opciones metodológicas son viables para abordar varios de los objetivos y preguntas de investigación planteadas en éste proyecto, aún hay varios detalles que deben resolverse.

**Las búsquedas de cadenas de texto pueden usarse para corrección o normalización de campos con cadenas cortas y catálogos denidos, como nombres y lugares geográficos, pero con cadenas de texto largas, como los campos con texto libre, es necesario usar métodos fuzzy con representaciones vectoriales apropiadas, como las obtenidas con LLMs contextuales como BERT, GPT o DeepSeek [33]**. En éste caso, es necesario un proceso de fine-tunning, ya que la información contenida en los registros de interés es de dominios muy particulares. Esto aplica también para el proceso de NER.

Otro aspecto importante es la **presencia de datos faltantes, lo que implica la formulación de metodologías que sean robustas a la pérdida de información que surge en las diferentes fuentes de información relacionadas con los registros que se desean integrar**. En las secciones siguientes se describen las metodologías propuestas en éste proyecto para resolver éstas problemáticas.

## Preguntas de Investigación

Trabajaremos sobre el problema de **ligado de registros**, en el cual **se tienen dos o más bases de datos que, se presume, contienen información sobre algunas entidades en común**. Por ejemplo, pueden ser dos bases de datos sobre personas: una sobre pacientes admitidos a un servicio médico y otra sobre personas que han recibido algún tratamiento.
En este ejemplo, **ambas bases de datos hablan sobre personas**, y una pregunta que surge es: **¿se pueden unificar los datos de ambas bases de datos para tener una visión más completa sobre las personas que figuren en ambas?**

El ligado de registros es fundamental para unir bases de datos (cross-referencing o joining, en inglés). En el caso de una base de datos diseñada originalmente para soportar esta operación, **se asignan identificadores únicos a cada entidad, lo que vuelve el problema trivial**. Asímismo, **cuando hay algún identificador en común en bases de datos, tales como CURP, DOI o algún otro, es muy fácil ligar registros entre ellas**. Sin embargo,estos casos triviales aparecen pocas veces en la realidad. **En la mayoría de los casos, las bases de datos no tienen identificadores confiables**, ya sea por problemas de implementación o, simplemente, porque no es posible identificar a las entidades en todos los casos.

Existen dos ejemplos paradigmáticos de este, el primero son **personas fallecidas que son ingresadas en un servicio médico forense sin identificación alguna**. El segundo ejemplo es **cuando una de las bases de datos no está estructurada alrededor de las entidades**, por ejemplo cuando se trata de una base de datos de documentos en donde se incluyen textos en lenguaje natural que mencionan a algunas de las entidades pero donde no se contempló nunca utilizar identificadores. Este segundo es el caso de las carpetas de investigación y otros documentos judiciales, en donde a cada documento se le asigna, quizás, un identificador como número de expediente o similar, pero a las personas, lugares, eventos u organizaciones que ahí se mencionan no se les asignan identificadores que permita su ligado con otras bases de datos.

## Pregunta central:

**¿Cómo aprovechar las capacidades de comprensión semántica de los LLMs para mejorar el proceso de ligado de registros, incluyendo el ligado entre textos y catálogos estructurados, considerando tanto las limitaciones técnicas (e.g., tamaño de contexto) como las características propias de los datos gubernamentales (e.g., errores, ambigüedad, sensibilidad y acceso restringido)?**

De acuerdo a la metodología descrita abajo, esta pregunta se abordará respondiendo cada una de las siguientes preguntas:

1. **¿Cómo aprovechar las capacidades de entendimiento de lenguaje de los LLMs para el ligado de entidades, tomando en cuenta las limitaciones en tamaño de contexto de estos a la vez que se aprovecha toda la información contenida en catálogos?**
    Con esto nos referimos a encontrar coincidencias en textos, de entidades contenidas en algún catálogo. Un ejemplo de esto sería determinar en un párrafo que menciona al municipio de *"San Pedro"*, a cuál de los muchos municipios que incluyen este string en su nombre se refiere. Para ello podría tomarse en cuenta información adicional como el estado o región del que habla el párrafo o donde fue producido el documento, y contrastar ésta con la información jerárquica del catálogo de municipios del INEGI. De esta forma, si el texto habla de una persona que vive en el Área Metropolitana de Monterrey, será posible saber que se trata de *"San Pedro Garza García"*.

2. **¿Cómo aprender automáticamente medidas de similitud entre representaciones vectoriales (embeddings) de datos, cuando estos estén representados de formas distintas en dos bases de datos?**

    Con esto nos referimos a encontrar formas de representar datos semánticamente comparables, que trascienda las representaciones o serializaciones que de éstos se hagan. Por ejemplo, si un registro en una base de datos se refiere a la fecha de nacimiento de una persona como el 15 de enero de 1990, y otro registro habla de una persona de 35 años de edad que ingresó el 4 de abril de 2025 a un servició médico, es posible que ambos registros hablen de la misma persona. Transformar estas maneras de almacenar información en una misma representación vectorial, nos permitirá aprovechar las técnicas de aprendizaje automático para encontrar funciones que asignen scores de similitud a dichos datos, y así producir listas de **posibles coincidencias a ser verificadas por humanos**.

3. **¿Cómo tomar en cuenta, durante el ligado de entidades, la no-independencia entre distintos campos?**

    Mientras que los métodos clásicos de ligado de entidades suponen una independencia entre los distintos campos y, por lo tanto, combinan los scores de similitud de distintos campos para construir un score de similitud entre registros, en este proyecto **se tomará en cuenta la interdependencia entre los valores de distintos campos y la necesidad de combinar estos de manera no-lineal**.

    Para ejemplificar esta no-independencia considerense tres registros:
    - **El registro A** que habla de una persona de sexo masculino con nombre Alejandro Mendez,
    - **El registro B** que habla de una persona de sexo femenino con nombre Alejandra Mendez, y
    - **El registro C** que habla de una persona de sexo masculino con nombre Alejando Mendes.

    Si consideramos únicamente la similitud de cadenas de texto, los registros A y B están a distancia uno, sin embargo, la naturaleza de la distribución de nombres, y su dependencia con el sexo de las personas, hace muy probable que trate de personas distintas. Por otro lado, los registros B y C que están a distancia dos en términos de nombres, se refieren con mayor probabilidad a la misma persona, dada la distribución de nombres y apellidos, así como de los errores ortográficos más comunes.

    **Para considerar dicha interdependencia, se plantea el uso de clasificadores estadísticos basados en redes neuronales artificiales, que actuen no sobre los scores de similitud de cada campo, sino sobre una concatenación de las representaciones vectoriales aprendidas por cada campo que, como se mencionó anteriormente, deberán ser tolerantes a formas distintas de almacenamiento de información en cada base de datos**.

En todas estas hay que tomar en consideración:

**(a)** Existen errores y datos faltantes de origen, por lo que es necesario tomar en cuenta no solo la similitud sino la certidumbre. Por ejemplo, en una base de datos algunos registros tienen errores de ortografía, o algunos registros tienen nombre y dos apellidos, y otros registros tienen solo el nombre propio. Esto se tiene que tomar en cuenta al calcular el score de certidumbre (normalización)

**(b)** Escala: algunos conjuntos de datos son muy grandes y se necesitan algoritmos de similaridad de varias granularidades para abordarlas

**(c)** Es necesario reconciliar los objetivos con las restricciones propias de datos de gobierno, sensibilidad, restricciónes de acceso, etc. tomando en cuenta que la responsabilidad final de los resultados debe recaer en las personas

## Pertinencia

La creciente disponibilidad de datos en el ámbito gubernamental, particularmente en los sectores de salud y seguridad, ha acentuado la **necesidad de desarrollar métodos robustos para la vinculación de registros que hacen referencia a las mismas entidades en diferentes bases de datos**. Este proceso, conocido como ligado de registros o *entity resolution*, es fundamental para construir representaciones integrales de personas a partir
de fuentes heterogéneas.

La pertinencia de este problema es fundamental para las instituciones públicas mexicanas, muchas de las cuales se encuentran en procesos de modernización tecnológica, enfrentando de forma simultánea retos de interoperabilidad interna y externa. Las bases de datos administrativas varían considerablemente en su diseño, calidad y criterios de captura. Es común que existan errores humanos en la recolección de datos, incluyendo omisiones, errores de ortografía, recodificaciones manuales, y divergencias respecto a catálogos oficiales. Por ejemplo, la misma localidad puede aparecer como “Cuajimalpa, México” en una base y como “Cuajimalpa de Morelos, Ciudad de México” en otra, generando ambigüedades semánticas. A ello se suma la confusión frecuente entre “México”, la cual puede referirse al Estado de México, el nombre informal de la zona metropolitana o el país.

Además de la variabilidad lingüística, existe una alta prevalencia de registros incompletos. En el caso del archivo Hístorico del Servicio Médico Forense (SEMEFO), el 17.21% de los registros corresponden a personas “desconocidas”, y en una proporción aún mayor se omite información clave como el segundo apellido, la edad precisa o incluso el nombre propio. Este grado de incompletitud compromete los métodos tradicionales de vinculación, e impone la necesidad de incorporar modelos que integren incertidumbre y tolerancia al error como parte del diseño algorítmico.

Las dificultades aumentan en casos con presencia de homónimos, duplicados, reasignación de identificadores únicos por procesos internos, o cuando el mismo individuo tiene múltiples ingresos en el mismo sistema, como es el caso de los ingresos hospitalarios. Estas condiciones hacen inviable el uso de medidas simples de comparación entre cadenas de texto, pues no reflejan adecuadamente la semántica o el contexto. Por ejemplo, una comparación directa entre “Alejando”, “Alejandra” y “Aleijandr0” puede producir valores engañosos si usan métricas lineales y no se toman en cuenta atributos como el sexo registrado, que introduce una **dependencia entre campos**.
El contar con métodos robustos y eficientes para integrar registros no solo coadyuva al fortalecimiento de la toma de decisiones basada en evidencia, sino también es necesaria para garantizar el derecho a la atención médica o a la justicia. En salud, una vinculación incorrecta puede derivar en fallas en el tratamiento de un paciente. Pero además, las instituciones de salud públicas requieren conjuntos de datos confiables para establecer políticas a nivel poblacional, cómo es el caso de la epidemiología o campañas de salud.
**En el ámbito forense y de desapariciones, la correcta correspondencia entre registros puede significar la localización o no de una persona desaparecida, por lo que los errores tienen consecuencias humanas directas**.

**Dado que se trata de conjuntos de datos masivos, con millones de registros en algunos casos, los métodos propuestos deben ser escalables y computacionalmente eficientes**. Sin embargo, esta necesidad de eficiencia no puede desvincularse de la incorporación del juicio humano en el proceso: **los modelos deben estar diseñados para facilitar el trabajo de revisión por parte de operadores institucionales, ofreciendo estimaciones de similitud junto con medidas explícitas de incertidumbre que permitan priorizar los casos con mayor ambigüedad**.

Abordar este problema desde un enfoque científico requiere el uso de tecnologías de vanguardia como *fuzzy matching*, *LLMs*, *embeddings* y estrategias de aprendizaje, todo ello bajo estrictos criterios de transparencia, auditabilidad y respeto a los marcos normativos aplicables al tratamiento de datos personales.

Este proyecto no solo busca resolver un problema técnico de alta complejidad, sino también **contribuir al avance del conocimiento en las áreas de inteligencia artificial, gestión de datos y ciencia computacional**. En particular, se espera que los resultados obtenidos puedan integrarse en pipelines más amplios de construcción de grafos de conocimiento, con impacto directo en el fortalecimiento de las capacidades institucionales del Estado mexicano y en el desarrollo de herramientas replicables en contextos internacionales comparables.

## Metodología

Este proyecto desarrollará nuevos métodos para el ligado de registros entre bases de datos disimiles, siendo estos tolerantes a la diversidad de formas de almacenar la información en dichas bases, a las cantidades grandes de registros presentes en algunas de estas, y a la sensibilidad de los datos ahí contenidos. Por ello, diferenciamos a continuación entre la *metodología de trabajo*, y la *metodología técnica*.

Para lograr los objetivos del proyecto, se plantea una *metodología de trabajo* en donde las personas investigadoras no necesariamente tengan acceso a las bases de datos, sino que las instituciones colaboradoras puedan poner a prueba y evaluar el resultado de los métodos conforme estos vayan desarrollándose. En particular, siendo conscientes de que no será posible tener acceso directamente a datos de la Fiscalia General de Justicia de la Ciudad de México (FJG-CDMX), se prevé que el desarrollo de software pueda ser desplegado en sucesivas versiones en los sistemas de dicha dependencia, y operado por personas autorizadas para su puesta a prueba y evaluación.

En particular, **en las fases iniciales del proyecto** se plantea el uso de datos públicamente disponibles, o que obren en poder de las instituciones participantes. Entre éstos se destacan los **catálogos del INEGI, un conjunto de mil fichas pertenecientes al catálogo de la extinta Dirección Federal de Seguridad que han sido transcritas a mano y contienen menciones de personas, eventos y organizaciones**, así como **bases de datos del INER (institución participante en el proyecto) sobre admisiones de pacientes a distintos servicios**.
Con estos datos se entrenaran una serie de redes neuronales artificiales como se describe más adelante, que podrán ser desplegadas en un contenedor Docker en las instalaciones de otras instituciones, permitiendo evaluar con datos que deben permanecer en éstas. El resultado de dichas evaluaciones habrá de ser devuelto a las personas investigadoras en términos de listas de parejas de vectores y las evaluaciones que de éstas hagan los expertos, respetando así la regulación aplicable en materia de reserva de datos y privacidad.

Desde el **punto de vista técnico**, se plantea una **metodología** de tres pasos.

**Primero**, se identificarán **grupos de variables, denominados bloques, que sean semánticamente equivalentes, en tanto que contengan información comparable** aunque ésta se encuentre descrita a lo largo de distintos números y tipos de campos. Como ejemplo, el bloque que se refiere a la fecha de nacimiento puede ser descrito como un solo campo de tipo date en una sola base de datos, y como la combinación de un campo de edad (integer) y fecha de registro (datetime) en otra base de datos. La identificación de dichos bloques se hará de manera semiautomática, con ayuda de sistemas existentes basados en comparaciones de descripciones textuales de los campos[22]. **A los campos de una base de datos que corresponden a un bloque, le llamamos el *soporte* de dicho bloque en dicha base de datos**. De esta forma **un mismo bloque tendrá más de un soporte, uno por base de datos**.

La pregunta 1 de investigación mencionada anteriormente será abordada considerando bloques que tengan, por un lado, un soporte en un campo dotado de información jerárquica (un catálogo de municipios por estado, por ejemplo) y, por otro lado, un soporte en un campo de texto libre (una descripción de hechos en una denuncia, por ejemplo).

**Segundo**, habiendo identificado los bloques, se procederá a **entrenar redes neuronales de tipo encoder para la generación de representaciones vectoriales (embeddings) para cada soporte que permitan la identificación de registros similares marcados en un conjunto de datos de entrenamiento**. Este entrenamiento se hará bloque por bloque, aprovechando así la posibilidad de anotar datos de entrenamiento para un solo bloque provenientes de distintas bases de datos, reduciendo de esta manera el volúmen de datos necesarios. **La generación de representaciones vectoriales comunes para hacer comparables distintos tipos de datos se ha abordado de varias formas, incluyendo la verbalización de los datos a lenguaje natural [26] y el aprendizaje de representaciones multimodales [27]; este planteamiento se relaciona directamente con la literatura sobre aprendizaje multimodal y fusión de datos [23][24][25].** El entrenamiento de estas redes neuronales y las representaciones vectoriales resultantes constituyen el abordaje de la pregunta de investigación 2. Ver Figura 1 para una descripción gráfica.


**[Figura 1] Se generarán representaciones vectoriales por cada soporte, a partir de las cuales se puede calcular una similaridad por bloque. Estas representaciones vectoriales a su vez se utilizarán para entrenar los algoritmos de ligado de registros.**

Para un bloque dado $b$, se computarán pues funciones de representación diversas, una por soporte. Cada una de éstas, aprendida por medio de una red neuronal como se detalla abajo, tendrá un dominio distinto (dependiendo de los datos que para ese bloque existan en esa base de datos) pero todas tendrán el mismo rango: el espacio de embeddings del bloque. En este espacio será entonces posible definir funciones de similitud que produzcan, además un score, entre $0$ y $1$, de similitud, una score de certeza que dependa de la cantidad y calidad de los datos cuyos embeddings se computan.

Estas funciones de representación por bloques serán aprendidas como redes neuronales feedforward que sean entrenadas para identificar si se refieren a la misma entidad (persona, paciente, etc) o que proviene de entidades diferentes. Para esto, se utiliza una función de pérdida cuya minimización lleva al ajuste de las entradas de los embeddings de tal forma que las representaciones de entidades similares tienen una distancia muy pequeña y las de entidades diferentes tendrán distancias grandes. La función de pérdida para aprender dichas representación será, por ejemplo, Contrastive Loss function [32]:

$$ L = (1-y) \frac{1}{2} d(E_i, E_j)^2 + y \frac{1}{2} \{max[0, \delta - d(E_i, E_j)]\}^2 $$


donde $d$ es la distancia Euclidiana, $y$ es una variable que indica si las entidades son diferentes y $\delta$ es un hiperparámetro que indica el márgen de distancia entre entidades distintas. **Las entidades similares y diferentes, se seleccionan del conjunto de datos de entrenamiento, por lo que es necesario tener un conjunto de datos previamente etiquetado**.

El esquema de esta red neuronal se muestra en la **Figura 2**, donde los Inputs corresponden a la fusión mostrada en la **Figura 1**.

**[Figura 2] Red neuronal siamesa para aprender embeddings que representen la distancia entre registros similares y registros diferentes.**

Esta estrategia para obtener representaciones por bloques, es lo suficientemente **flexible para incluir distintos tipos de variables que esperaríamos encontrar en los registros (textuales, fechas, categóricas o continuas), incluyendo su representación individual, ya que el encoder de cada campo es independiente. Pueden usarse desde embeddings con LLMs, hasta métodos basados en componentes principales, análisis de correspondencias múltiple u otros, dependiendo de la naturaleza de las variables**. **En caso de tener datos faltantes en algún campo de soporte, nuestra metodología contempla usar estrategias de imputación de modalidades [30][31] (valores cero o promedios, por ejemplo), o más sofisticadas, como las basadas en atención *intra-modality* o *inter-modality* [28], dependiendo de la cantidad de campos faltantes, viéndose esto reflejado también en el correspondiente score de certeza.

**Tercero**, una vez obtenidas las representaciones por cada bloque de información de cada base de datos, **el siguiente paso consiste en la inferencia para nuevos registros**, la cuál puede realizarse en varios niveles.
Supongamos dos registros $x$ y $y$ de dos bases de datos (posiblemente) distintas $A$ y $B$, y sean $b_1, b_2,..b_k$ los bloques que ambos tienen en común.
Llamemos $V_{b,A}$ a la función (aprendida usando una red neuronal artificial de tipo encoder) que genera una representación vectorial del soporte del bloque en la base de datos $A$, a la representación del soporte del mismo bloque en la base de datos $B$. Una medida de similaridad global entre los registros y está dada por

$$d(x,y) = \dfrac{\sum_{b \in \{b_1, b_2, \ldots, b_k\}} \alpha_{b} \, d_b \left( V_{b, A}(x), V_{b,B}(y) \right)}{\sum_{b \in \{b_1, b_2, \ldots, b_k\}} \alpha_{b}}$$


donde $d_b$ es una función de similaridad entre representaciones vectoriales (embeddings) del bloque $b$, y  $\alpha_b$ es un peso asignado a dicho bloque, que puede ser $0$ cuando alguno de los dos registros no tiene datos suficientes en el bloque, o cuando la fiabilidad de dicho embedding se estima baja.

**Usando esta distancia se pueden obtener los registros similares a un $x$ dado usando los $K-$ vecinos más cercanos en el espacio de los embeddings de una base de datos de referencia**. Ordenando las distancias obtenidas con todos los registros de una base de datos, se pueden obtener parejas de registros candidatos a ser ligados con $x$ para su posterior revisión por parte de humanos.

Es importante resaltar que los métodos a emplearse deben ser aplicables en las condiciones específicas de este proyecto, por lo que deberán considerar:

1. **La privacidad o reserva de los datos.** Al trabajar sistemas que produzcan
representaciones vectoriales (embeddings) por cada soporte, se incrementa la
posibilidad de transferir el aprendizaje entre un conjunto de datos totalmente
accesible o público, y un conjunto de datos al que no se tiene acceso. De esta
forma, una red neuronal que produce representaciones vectoriales de nombres,
por ejemplo, puede ser entrenada en un conjunto público, y después ejecutada en las instalaciones de FJG-CDMX, sin necesidad de utilizar sus datos para el
entrenamiento.
Adicionalmente, las evaluaciones hechas por personal de la Fiscalía, junto con los valores de capas intermedias de las redes neuronales que habrán de procesar las representaciones vectoriales arriba mencionadas, podrán ser utilizadas para refinar los algoritmos de ligado, sin comprometer los valores exactos de los datos, utilizando la técnica de auto-encoders privados.[29]

2. **El gran volúmen de datos.** Este proyecto tiene la particularidad de que, habiendo entrenado todos los algoritmos de aprendizaje de máquina previstos, aún hay una gran cantidad de datos por ser procesados por éstos en modo inferencia.
Para dar dimensión a esto, en la CDMX se abrieron 234,276 carpetas de investigación , sólo en el año de 2023. La mitad de éstas tiene más de 1001 páginas dando un aproximado de más de 30 millones de páginas de texto a ser analizadas.
Para poder procesar estos volúmenes de información, los sistemas de software que habrán de resultar de este proyecto deberán ser escalables horizontalmente, para lo cual se prepararán en contenedores listos para ser desplegados en nubes públicas o privadas. Así mismo, para el procesamiento de los datos que sean públicamente accesibles, el equipo de cómputo a adquirirse complementara el ya desplegado en la institución beneficiaria para facilitar su procesamiento en tiempos razonables. Finalmente, la arquitectura modular de las redes neuronales, y el uso de versiones cuantizadas de modelos, permitirá reducir la cantidad de procesamiento necesario para ejecutar los modelos.
3. **La ausencia de datos.** Una característica de las bases de datos en el mundo real es la falta de datos en muchos registros. Las razones para esto son variadas, pero en el presente proyecto no se plantea solucionar este problema, sino trabajar asumiendo esta característica de los datos. En particular se adoptarán estrategias conocidas para generar representaciones vectoriales de bases de datos con valores faltantes [30,31]. En este sentido, la estrategía basada en bloques descrita anteriormente, permite el entrenamiento de la representación de cada grupo utilizando datos de más alta calidad y transferir posteriormente el entrenamiento a datos con bloques faltantes.

## Resultados esperados

Los resultados están relacionados con las tres preguntas de investigación. Se plantea un trabajo iterativo donde, conforme se vayan resolviendo las preguntas, se vayan implementando la solución en sistemas que puedan ser puestos a prueba por las instituciones participantes. Así mismo, la resolución de las preguntas irá ligada con la publicación académica de resultados, la formación de alumnos, y la generación de competencias en las instituciones para su máximo aprovechamiento.

Concretamente, se espera que al final del proyecto se entregarán los siguientes resultados. Estos se presentan a continuación divididos entre aquellos de carácter académico, en tanto que avanzan el estado de la técnica, y aquellos de carácter institucional que conllevan a una mejora en los procesos de las instituciones involucradas en el proyecto como casos de uso.

Resultados esperados de carácter académico:

- a) 3 Artículos de investigación figurando como memorias de conferencias internacionales especializadas en áreas de aprendizaje automático y bases de datos.
- b) Formación y titulación de alumnos de licenciatura y uno de maestría.
- c) Publicación de base de datos anotada de ligado de entidades para su reutilización por parte de otros investigadores.
- d) Publicación del código fuente de las herramientas de software desarrolladas.
- e) Reporte de exploración inicial sobre métodos de comparación usando gráfos de conocimiento multimodales asociadas a una persona.

Resultados esperados de carácter institucional:

- f) Artículo de difusión, enfocado a todo público, para concientizar sobre el estado actual de la integración de datos públicos y sus posibles abordajes.
- g) Herramienta de software para el uso del INER y de la FGJ-CDMX. Esta se desarrollará iterativamente conforme se avance en los tres pasos de la metodología, y se realizarán ajustes a los modelos de redes neuronales según la retroalimentación de las personas expertas. Será desplegada en los ambientes relevantes.
- h) Videos y cursos de capacitación para el uso de la herramienta, así como para la adaptación de la misma para otros casos de uso.

## Riesgos para el proyecto y sus mitigaciones

La ejecución del presente proyecto implica diversos desafíos técnicos y operativos que podrían afectar el desarrollo oportuno y exitoso de sus objetivos.
A continuación, se describen los principales riesgos identificados, junto con las estrategias de mitigación propuestas para cada uno:

1. Requerimientos computacionales elevados. El uso de modelos grandes de lenguaje (LLMs) y el procesamiento de grandes volúmenes de datos implican una alta demanda de recursos computacionales, particularmente de unidades de procesamiento gráfico (GPU).
    a. Actualmente, se cuenta con infraestructura computacional con capacidades adecuadas para la primera fase del proyecto.
    b. En el presupuesto se considera la compra de un GPU adicional el primer año del proyecto.
    c. Para los años subsiguientes, en caso de que la escala de los modelos lo requiera, se contempla el uso de servicios en la nube.

2. Acceso limitado o revocable a bases de datos sensibles. Dado que el trabajo se basa en el análisis de registros reales, la disponibilidad continua de los datos proporcionados por las instituciones colaboradoras es crucial. Existe el riesgo de que se revoque o retrase el acceso a los datos, particularmente en el caso de los registros de la Fiscalía General de Justicia de la Ciudad de México.

    a. El Instituto Nacional de Enfermedades Respiratorias (INER), como institución colaboradora, dará acceso a varios de sus conjuntos de datos.
    b. El Histórico del SEMEFO es público por medio del INAI, garantizando acceso inicial a conjuntos de datos forenses.
    c. En caso de que el acceso a datos de la FGJ-CDMX sea restringido o revocado, se prevé la utilización de datos abiertos disponibles en el portal `datos.gob.mx`, así como de otros conjuntos relevantes como el Padrón del IMSS y RNPDNO 2023.
    d. Se contempla también la recolección estructurada (crawling) de bases de datos públicas o históricas (como los archivos de la represión) para análisis comparativos.
    e. Se mantendrá una constante vinculación con la FGJ-CdMx.

3. Insuficiencia de datos anotados para entrenamiento de modelos de aprendizaje automático. Los métodos de aprendizaje profundo para estimar similitud en espacios de embeddings requieren conjuntos de datos suficientemente grandes y representativos para su entrenamiento.

    a. El proyecto contempla la construcción de una base de datos anotada a partir de los registros obtenidos, lo cual permitirá contar con ejemplos positivos y negativos de vinculación entre entidades.

4. Manejo de datos sensibles. El carácter sensible de los datos clínicos, forenses y administrativos utilizados en el proyecto requiere cuidados especiales en su tratamiento y análisis.

    a. Los participantes firmarán acuerdos de confidencialidad para acceder a los datos sensibles.
    b. Se diseñará una metodología rigurosa para el manejo seguro de datos personales y sensibles.
    c. Se observarán los marcos normativos nacionales e institucionales vigentes.

## Impacto social

El proyecto tiene el potencial de generar impactos sustantivos en el fortalecimiento de las capacidades institucionales de sectores estratégicos como salud y justicia, así como en la calidad de vida de las personas al mejorar los procesos de toma de decisiones basadas en datos.

En el ámbito de la salud, instituciones como el Instituto Nacional de Enfermedades Respiratorias (INER) se encuentran en medio de una transformación digital que implica la homologación e integración de múltiples sistemas de información. La existencia de expedientes clínicos electrónicos coexistiendo con registros en papel y sistemas heterogéneos genera duplicidades, inconsistencias y dificultades para dar seguimiento a la trayectoria clínica de un paciente. Este problema se agrava en situaciones de emergencia, como la pandemia por COVID-19, donde errores humanos en la captura de datos son frecuentes, ya que se prioriza la atención de los pacientes. El desarrollo de métodos avanzados de ligado de registros permitirá reducir drásticamente los tiempos de procesamiento de información y mejorar la calidad y oportunidad de la atención al paciente.

Asimismo, la adecuada integración de datos es indispensable para que las instituciones de salud cumplan sus funciones sustantivas a nivel poblacional. Una base de datos homogénea y bien vinculada facilita el acceso oportuno a información crítica para la vigilancia epidemiológica, la planificación de recursos y el análisis en economía de la salud. La capacidad de identificar rápidamente patrones y tendencias a partir de datos integrados puede traducirse en políticas públicas más efectivas y en una mejor respuesta a emergencias sanitarias.

En el ámbito de justicia y seguridad, la Fiscalía General de Justicia (FGJ) enfrenta retos similares. La mala homologación de registros disminuye la calidad de los datos y limita la capacidad de reutilización del trabajo de los agentes del ministerio público y peritos. Un sistema robusto de homologación de datos permite además la colaboración efectiva e interoperabilidad entre instituciones, al permitir la vinculación de carpetas de investigación con bases de datos externas, como las del INCIFO o registros de personas desaparecidas, habilitando el cruce de información que de otro modo permanecería fragmentada. Esta capacidad no solo impactaría en la resolución de casos individuales, sino también permitiría abordar problemas estructurales.

El impacto del proyecto trasciende a las instituciones colaboradoras iniciales, ya que el problema del ligado de registros es un desafío común en múltiples sectores y un paso necesario en muchos pipelines de ciencia de datos aplicada. Las herramientas y las metodologías desarrolladas podrán ser transferidas y adaptadas a otras dependencias gubernamentales, organismos internacionales y sectores privados interesados en mejorar sus procesos de gestión de datos. En resumen, este proyecto no solo contribuirá al avance científico en inteligencia artificial, sino que tendrá un impacto directo en la eficiencia institucional, la mejora de servicios públicos y, en última instancia, en el bienestar de la población.

## Bibliografía
[1] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. Introduction to Algorithms, Third Edition. MIT Press and McGraw- Hill, 2009. ISBN 0-262-03293-7. Chapter 32: String Matching, pp. 985– 1013.

[2] Paul E. Black, "string matching", in Dictionary of Algorithms and Data Structures [online], Paul E. Black, ed. 2 November 2020. (accessedApril 22, 2025) Available from: https://www.nist.gov/dads/HTML/ stringMatching.html

[3] Knuth D.E., Morris Jr. J.H., Pratt V.R. Fast pattern matching in strings. SIAM J. Comput., 6 (1977), p. 323.

[4] Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics Doklady, 10(8), 707–710.

[5] Damerau, F. J. (1964). A technique for computer detection and correction of spelling errors. Communications of the ACM, 7(3), 171–176.

[6] Jaro, M. A. (1989). Advances in record-linkage methodology as applied to matching the 1985 census of Tampa, Florida. Journal of the American Statistical Association, 84(406), 414–420.

[7] Winkler, W. E. (1990). String comparator metrics and enhanced decision rules in the Fellegi-Sunter model of record linkage. Proceedings of the Section on Survey Research Methods, 354–359.

[8] Ukkonen, E. (1992). Approximate string-matching with q-grams and maximal matches. Theoretical Computer Science, 92(1), 191–211.

[9] Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513–523.

[10] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. 2003. A neural probabilistic language model. J. Mach. Learn. Res. 3, 1137–1155.

[11] Goldberg, Y. (2017). Neural Network Methods for Natural Language Processing. Morgan & Claypool.

[12] Ruder, S. (2018). A Survey of Cross-lingual Word Embedding Models. arXiv:1706.04902.

[13] Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781.

[14] Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation. EMNLP.

[15] Bojanowski, P., et al. (2016). Enriching Word Vectors with Subword Information. arXiv:1607.04606.

[16] Peters, M., et al. (2018). Deep Contextualized Word Representations. NAACL.

[17] Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805.

[18] Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-training. OpenAI.

[19] Jing Li, Aixin Sun, Jianglei Han, and Chenliang Li. 2022. A Survey on Deep Learning for Named Entity Recognition. IEEE Trans. on Knowl. and Data Eng. 34, 50–70. https://doi.org/10.1109/TKDE.2020.2981314

[20] Dunn, H. L. (1946). "Record Linkage". American Journal of Public Health. 36 (12): pp. 1412–1416. doi:10.2105/AJPH.36.12.1412.

[21] Fellegi, I.; Sunter, A. (1969). "A Theory for Record Linkage" (PDF). Journal of the American Statistical Association. 64 (328): pp. 1183–1210. doi:10.2307/2286061.

[22] Zhang, Y., Floratou, A., Cahoon, J., Krishnan, S., Müller, A. C., Banda, D., ... & Patel, J. M. (2023, April). Schema matching using pre-trained language models. In 2023 IEEE 39th International Conference on Data Engineering (ICDE) (pp. 1558-1571). IEEE.

[23] Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2018). Multimodal Machine Learning: A Survey and Taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).

[24] D. Ramachandram and G. W. Taylor, "Deep Multimodal Learning: A Survey on Recent Advances and Trends," in IEEE Signal Processing Magazine, vol. 34, no. 6, pp. 96-108, Nov. 2017, doi: 10.1109/ MSP.2017.2738401.

[25] D. Lahat, T. Adali and C. Jutten, "Multimodal Data Fusion: An Overview of Methods, Challenges, and Prospects," in Proceedings of the IEEE, vol. 103, no. 9, pp. 1449-1477, Sept. 2015, doi: 10.1109/ JPROC.2015.2460697.

[26] Cappuzzo, R., Papotti, P., & Thirumuruganathan, S. (2020, June). Creating embeddings of heterogeneous relational datasets for data integration tasks. In Proceedings of the 2020 ACM SIGMOD international conference on management of data (pp. 1335-1349).

[27] Jiang, T., Song, M., Zhang, Z., Huang, H., Deng, W., Sun, F., ... & Zhuang, F. (2024). E5-v: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580.

[28] Wu, R., Wang, H., Chen, H., & Carneiro, G. (2024). Deep Multimodal Learning with Missing Modality: A Survey. https://arxiv.org/abs/ 2409.07825

[29] Alguliyev, R. M., Aliguliyev, R. M., & Abdullayeva, F. J. (2019). Privacy-preserving deep learning algorithm for big personal data analysis. Journal of Industrial Information Integration, 15, 1-14.

[30] Ong, T. C., Mannino, M. V., Schilling, L. M., & Kahn, M. G. (2014). Improving record linkage performance in the presence of missing linkage data. Journal of biomedical informatics, 52, 43-54.

[31] Ghorbani, A., & Zou, J. Y. (2018, October). Embedding for informative missingness: Deep learning with incomplete data. In 2018 56th Annual Allerton Conference on Communication, Control, and Computing (Allerton) (pp. 437-445). IEEE.

[32] R. Hadsell, S. Chopra and Y. LeCun, "Dimensionality Reduction by Learning an Invariant Mapping," 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06), New York, NY, USA, 2006, pp. 1735-1742, doi: 10.1109/CVPR.2006.100.

[33] DeepSeek. (2024). DeepSeek Chat (Version V3) [Large language model]. https://www.deepseek.com


[34] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP).

[35] Li, J., et al. (2020). Deep Entity Matching with Pre-Trained Language Models. Proceedings of the VLDB Endowment, 14(1), 50-60.
</div>
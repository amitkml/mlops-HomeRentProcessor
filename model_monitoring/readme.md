# Model Monitoring
This section aims to monitor ML models using Evidently, Grafana, Prometheus and Prefect. The scripts perform basic model monitoring that calculates and reports metrics and saves evidently report in the form of HTML file.

## Monitoring in different paradigms:

### Batch:
In Batch Models, we implement batch monitoring. We add some calculation block after step in the pipeline and run some checks to make sure that the model behaves as expected. In other words:
- Calculate performance metrics and health metrics
- Log the metrics in a SQL or NoSQL database
- Build a report

### Online models:
- In real-time served models, we may want to keep a closer live look on how the model performs. We add a service that pulls metrics and update the visuals in real time.

Sometimes, despite the model being online, we may want to monitor the model in Batch mode as well. As certain issues in the model may only manifest themselves over a longer timespan or larger dataset, such as Data Drift and Concept Drift.

## Monitoring our service
We want to monitor our previously deployed House rent predictor.  In particular, we want to monitor our Backend model. In this monitoring deployment, we want both Online monitoring via Prometheus and Grafana as well as Offline monitoring via EvidentlyAI:

![im](https://user-images.githubusercontent.com/24941662/181019189-132b1634-6e32-463d-a7b0-1d4b48e4726a.png)

# Industrial Equipment Monitoring and Predictive Maintenance

## Project Overview
This project focuses on monitoring industrial machines using IoT sensors to detect performance anomalies. The system routes sensor data to cloud platforms (AWS and Azure) for processing and visualization. Predictive maintenance is enabled through Azure AI, while Power BI and Grafana provide real-time performance dashboards.

## Objective
- Monitor industrial equipment using IoT sensors (vibration, temperature, pressure).  
- Route sensor data to cloud platforms via middleware.  
- Visualize real-time performance and equipment health status.  
- Implement predictive maintenance to reduce downtime and costs.  

## Technologies Used

### IoT Sensors - Simulation
- Vibration sensors  
- Temperature sensors  
- Pressure sensors  

### Middleware
- MQTT broker for message queuing  

### Cloud Platforms
- **AWS**:  
  - IoT Core for device management  
  - Timestream for time-series data storage  
  - Lambda for serverless data processing  

- **Azure**:  
  - Log Analytics for log management  
  - Monitor for performance tracking  
  - AI/ML models for predictive maintenance  

### Visualization Tools
- Power BI: Equipment health status dashboard  
- Grafana: Live equipment performance dashboard  

## Architecture Diagram
The architecture diagram illustrates the flow of data from IoT sensors to cloud platforms and visualization tools. Key components include:  
1. IoT sensors collecting performance data.  
2. MQTT broker routing data to AWS and Azure.  
3. AWS services (IoT Core, Timestream, Lambda) processing and storing data.  
4. Azure services (Log Analytics, Monitor, AI) analyzing data for predictive insights.  
5. Power BI and Grafana displaying real-time dashboards.  

## Outcome
- Reduced machine downtime through early anomaly detection.  
- Lowered maintenance costs by leveraging predictive insights.  
- Improved operational efficiency with real-time monitoring.  


## Setup Instructions
1. Deploy IoT sensors and configure MQTT broker.  
2. Set up AWS IoT Core and Azure Log Analytics for data ingestion.  
3. Configure Lambda functions for data processing.  
4. Import dashboards into Power BI and Grafana.  

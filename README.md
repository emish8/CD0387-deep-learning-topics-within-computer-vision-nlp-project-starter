# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

For this project I used the provided dog breed dataset.
### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

I used pretrained resnet18 model for this project. Pretrained models saves time. I chose resnet-18 specifically because it is a popular convolutional neural network architecture that was originally trained on the ImageNet dataset. The ImageNet dataset is a large-scale visual recognition challenge, which contains over 14 million labeled images belonging to 1,000 different classes. 
The other advantage of resnet is that it is designed to overcome the one common drawback of deep networks which is vanishing gradient.

i have used three parameters for hypertuning -
1. Learning rate
2. batch size
3. epochs

Here are there ranges, hyperparameter_ranges = {
                                   "lr": ContinuousParameter(0.001, 0.1),
                                   "batch-size": CategoricalParameter([16,32]),
                                   "epochs": IntegerParameter(2,4)
                                   }

Remember that your README should:
- Include a screenshot of completed training jobs
![hpo1](https://user-images.githubusercontent.com/83595196/225536788-3b6d075e-afb3-48d1-9b6f-e426e06af68b.JPG)
![hpo2](https://user-images.githubusercontent.com/83595196/225536828-d21ba310-bc2c-4b1f-b338-8b2d2dae0fae.JPG)

- Logs metrics during the training process
![image](https://user-images.githubusercontent.com/83595196/225538095-daf3556a-1fa3-4834-8f70-b7b3ae1301fe.png)
![image](https://user-images.githubusercontent.com/83595196/225538413-1993f97a-022a-487d-b7bf-7cb4fa6f4eb0.png)

- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

{'_tuning_objective_metric': '"average test loss"',
 'batch-size': '"32"',
 'epochs': '4',
 'lr': '0.03305079691259119',
 'sagemaker_container_log_level': '20',
 'sagemaker_estimator_class_name': '"PyTorch"',
 'sagemaker_estimator_module': '"sagemaker.pytorch.estimator"',
 'sagemaker_job_name': '"pytorch-training-2023-03-15-13-37-17-692"',
 'sagemaker_program': '"hpo.py"',
 'sagemaker_region': '"us-east-1"',
 'sagemaker_submit_directory': '"s3://sagemaker-us-east-1-706723451900/pytorch-training-2023-03-15-13-37-17-692/source/sourcedir.tar.gz"'}
 
 
## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.

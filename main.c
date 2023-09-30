#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define numberOfInputs 4
#define numberOfHiddenNeurons 2
#define numberOfInputNeurons 2
#define numberOfOutputNeurons 1


//Prototypes
double InitiliazeWeights();
void RandomStateSetter(int *array, int size);
double Sigmoid(double x);
double SigmoidDerivative(double x);
int main(){
    
    srand(time(0));
    
    //Declaration of learning rate and iteration(epoch) number
    int iterationNumber;
    double learningRate;
    
    //Declaration of artifical neural network neurons
    double inputLayerNeurons[numberOfInputNeurons];
    double hiddenLayerNeurons[numberOfHiddenNeurons];
    double outputLayerNeurons[numberOfOutputNeurons];
    
    //Declaration of bias and weights
    double hiddenLayerWeights[numberOfInputNeurons][numberOfHiddenNeurons];
    double outputLayerWeights[numberOfHiddenNeurons][numberOfOutputNeurons];
    double hiddenBias[numberOfInputNeurons];
    double outputBias[numberOfOutputNeurons];
    
    //Declaration of the training set input and outputs
    double featureVector[numberOfInputs][numberOfInputNeurons] = {{0.0,0.0},{1.0,0.0},{0.0,1.0},{1.0,1.0}};
    double label[numberOfInputs][numberOfOutputNeurons] = {{0.0f},{1.0f},{1.0f},{0.0f}};
    
    
    //Initilization of the weights for hidden layer
    for(int i=0; i<numberOfInputNeurons; i++){
        for(int j=0; j<numberOfHiddenNeurons; j++){
            hiddenLayerWeights[i][j] = InitiliazeWeights();
            printf("The initial value of weight between input %d and hidden neuron %d is: %f\n",i+1,j+1,hiddenLayerWeights[i][j]);
        }
        
    }
    printf("\n");
    //Intilization of the weights for output layer
    for(int i=0; i<numberOfHiddenNeurons; i++){
        for(int j=0; j<numberOfOutputNeurons; j++){
            outputLayerWeights[i][j] = InitiliazeWeights();
            printf("The initial value of weight between hidden %d and output neuron %d is: %f\n",i+1,j+1,outputLayerWeights[i][j]);
        }
    }
    
    //Feed Forward of Artifical Neural Network
    printf("\n Please enter the number of iterations: ");
    scanf("%d",&iterationNumber);
    printf("Pleasae enter the learning rate: ");
    scanf("%lf",&learningRate);
    
    //Since we will use Stochastic Gradient Descent, we need to randomize the order of the training set to maximize the convergence chance.
    int trainingSetDataOrder[] = {0,1,2,3};
    
    //Going through each iteration (epoch)
    for(int iterationNum = 0; iterationNum<iterationNumber; iterationNum++){
        
        RandomStateSetter(trainingSetDataOrder, numberOfInputs);
        for(int rowRandomNumber=0; rowRandomNumber<numberOfInputs; rowRandomNumber++){
            int rowNumber = trainingSetDataOrder[rowRandomNumber];
            
            //Activation of the hidden layer
            for(int hiddenIndex = 0; hiddenIndex<numberOfHiddenNeurons; hiddenIndex++){
                double weightedSum = hiddenBias[hiddenIndex];
                for(int inputIndex = 0; inputIndex<numberOfInputNeurons; inputIndex++){
                    weightedSum+=featureVector[rowNumber][inputIndex]*hiddenLayerWeights[inputIndex][hiddenIndex];
                }
                hiddenLayerNeurons[hiddenIndex] = Sigmoid(weightedSum);
            }
            
            //Activation of the output hidden layer
            
            for(int outputIndex = 0; outputIndex<numberOfOutputNeurons; outputIndex++){
                double weightedSum = outputBias[outputIndex];
                for(int hiddenIndex = 0; hiddenIndex<numberOfHiddenNeurons; hiddenIndex++){
                    weightedSum += hiddenLayerNeurons[hiddenIndex] * outputLayerWeights[hiddenIndex][outputIndex];
                }
                outputLayerNeurons[outputIndex] = Sigmoid(weightedSum);
            }
            
            printf("Input: %g %g    Output: %g  Expected Output: %g\n",
                   featureVector[rowNumber][0],featureVector[rowNumber][1],outputLayerNeurons[0],label[rowNumber][0]);
            
            //Back Propogation
            
            //Error and slope calculation by using Sthocastic Gradient Descent
            //Error calculation of the output
            double deltaErrorOfOutput[numberOfOutputNeurons];
            for(int outputIndex=0; outputIndex<numberOfOutputNeurons; outputIndex++){
                double error = (label[rowNumber][outputIndex]-outputLayerNeurons[outputIndex]);
                deltaErrorOfOutput[outputIndex] = error*SigmoidDerivative(outputLayerNeurons[outputIndex]);
            }
            
            double deltaErrorOfHidden[numberOfHiddenNeurons];
            for(int hiddenlayerIndex = 0; hiddenlayerIndex<numberOfHiddenNeurons; hiddenlayerIndex++){
                double error = 0.0f;
                for(int outputIndex = 0; outputIndex<numberOfOutputNeurons; outputIndex++){
                    error += deltaErrorOfOutput[outputIndex] * outputLayerWeights[hiddenlayerIndex][outputIndex];
                }
                deltaErrorOfHidden[hiddenlayerIndex] = error*SigmoidDerivative(hiddenLayerNeurons[hiddenlayerIndex]);
            }
            
            //Applying changes to the weights and biases for output layer
            
            for(int outputIndex = 0; outputIndex<numberOfOutputNeurons; outputIndex++){
                outputBias[outputIndex] += deltaErrorOfOutput[outputIndex]*learningRate;
                for(int hiddenIndex = 0; hiddenIndex<numberOfHiddenNeurons; hiddenIndex++){
                    outputLayerWeights[hiddenIndex][outputIndex] +=hiddenLayerNeurons[hiddenIndex]*deltaErrorOfOutput[outputIndex]*learningRate;
                }
            }
            
            //Applying changes to the weights and biases for hidden layer
            for(int hiddenIndex = 0; hiddenIndex<numberOfHiddenNeurons; hiddenIndex++){
                hiddenBias[hiddenIndex] += deltaErrorOfHidden[hiddenIndex]*learningRate;
                for(int inputIndex = 0; inputIndex<numberOfInputNeurons; inputIndex++){
                    hiddenLayerWeights[inputIndex][hiddenIndex] += featureVector[rowNumber][inputIndex]*deltaErrorOfHidden[hiddenIndex]*learningRate;
                }
            }
            
            
            
            
        }
    }
    printf("\n");
    fputs("Final Hidden Weights\n[ ", stdout);
    for(int j=0; j<numberOfHiddenNeurons; j++){
        fputs("[ ", stdout);
        for(int k=0; k<numberOfInputNeurons; k++){
            printf("%f ", hiddenLayerWeights[k][j]);
        }
        fputs("] ",stdout);
    }
    
    fputs( "]\nFinal Hidden Biases\n[",stdout);
    for(int j=0; j<numberOfHiddenNeurons; j++){
        printf("%f ",hiddenBias[j]);
    }
    fputs("]",stdout);
    
    printf("\n");
    fputs("Final Output Weights\n[ ", stdout);
    for(int j=0; j<numberOfOutputNeurons; j++){
        fputs("[ ", stdout);
        for(int k=0; k<numberOfHiddenNeurons; k++){
            printf("%f ", outputLayerWeights[k][j]);
        }
        fputs("] ",stdout);
    }
    
    fputs("]\nFinal Output Biases\n[ ", stdout);
    for(int j=0; j<numberOfOutputNeurons; j++){
        printf("%f ",outputBias[j]);
    }
    
    fputs("] \n",stdout);
    
    
    
    
    return 0;
}

double InitiliazeWeights(){
    
    return(((double)rand())/((double)RAND_MAX));
}

void RandomStateSetter(int *array, int size){
    
    if(size>1){
        int i;
        for(i=0; i<size-1; i++){
            int j = i+rand()/(RAND_MAX/(size-i)+1);
            int temp = array[j];
            array[j] = array[i];
            array[i] = temp;
        }
    }
}

double Sigmoid(double x){
    
    return 1/(1+exp(-x));
}

double SigmoidDerivative(double x){
    
    return x*(1-x);
}

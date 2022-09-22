import java.util.Scanner;
//import javax.lang.model.util.ElementScanner14;
import java.io.File; 
import java.io.FileWriter;
import java.io.FileNotFoundException;
import java.io.IOException;

/*
* Caden Lin 
* Dr. Nelson 
* Honors Advanced Topics in Computer Science: Neural Networks 
* A--B--C--D Network in Java with Backpropagation 
* April 15 2022
* 
* Implementation of a neural network that can be trained and solve problems based on a given truth table. 
* The network includes just two hidden layers, but the number of inputs, the 
* number of activations per hidden layer, and the number of outputs are not restricted. 
* The network is trained using backpropagation. 
* 
* Table of contents: 
* public static void main(String[] args) throws FileNotFoundException, IOException
* public Network() throws FileNotFoundException, IOException 
* public void memoryAlloc(boolean train)
* public void populate(Scanner scan, boolean train, boolean random) throws FileNotFoundException
* public void run()
* public void train() throws IOException
* private void implementNetworkRun(int index)
* private void implementNetworkTrain(int index)
* private double calcError()
* private double actFunc(double theta)
* private double actFuncDeriv(double theta)
* private double randomNum(double low, double high)
* private String generateOutputString(double[] input, double[] truth, double[] out)
* private void saveWeights() throws IOException
* private void printNetworkInfo() throws IOException
* 
* The network is set up using information from a config file specified from the command line. 
* If nothing is passed, "config.txt" is used as the config file. 
* The format of the config file must be as follows:
* Line 1: The action that the network will perform. Enter "run" to run the network with the
* pre-loaded weights, "train random" to train the network using random weights, or "train preload" 
* to train the network starting with the preloaded weights. If something else is detected, the network is run by default. 
* Line 2: The number of hidden layers. As this network is an ABCD network, this number must be 2. 
* The network will not work otherwise. 
* Line 3: Number of rows in truth table 
* Line 4: Number of inputs 
* Line 5: Number of activations per hidden layer
* Line 6: Number of outputs 
* Line 7: Value of lamda used 
* Line 8: Error threshold 
* Line 9: Maximum number of iterations allowed 
* Line 10: The max and min values, respectively, for the range of the random number generator for the weights.
* Line 11: The name of the weights file to use when using pre-loaded weights. 
* Line 12: The name of the output file to save training weights to. 
*
* The next lines are the inputs from the truth table. Each line corresponds to a row from the truth table.
* Each value in each row corresponds to an input node. 
*
* The next lines are the outputs from the truth table. Each line corresponds to a row from the truth table.
* Each value in each row corresponds to an output node. 
*
*/
public class Network
{
   private static double[][] inputs; 
   private static double[][] truth; 

   private static double min;
   private static double max;
   private static double lamda; 

   private static double error; 
   private static double errorThreshold; 
   private static int iterations; 
   private static int maxIterations;

   private static double[][] network; 
   private static double[] a;
   private static double[] jTheta;
   private static double[] kTheta;
   private static double[] F;

   private static double[][] mkWeights;
   private static double[][] kjWeights;
   private static double[][] jiWeights;
   private static double[] psiI;
   private static double[] psiJ; 
   private static double[] psiK; 

   private static int numLayers; 
   private static int numInputs; 
   private static int numHiddenLayers; 
   private static int[] activationsPerHiddenLayer; 
   private static int numOutputs; 
   private static int numRows; 

   private static boolean defaultFile;  
   private static String weightsFile;
   private static String outputFile; 

   private static double time; 

   /*
   * The main method
   * Runs or trains the network using instructions from the config file 
   * 
   * @param args arguments from the command line 
   * @throws FileNotFoundException throws a FileNotFoundException if the config file cannot be found 
   * @throws IOException throws an IOException if the output file that contains the weights 
   * from training cannot be written or overridden
   */
  public static void main(String[] args) throws FileNotFoundException, IOException
  {
   String configFile = ""; 
   if (args.length > 0)
   {
      configFile = args[0];
   }
   else 
   {
      configFile = "src/testing_config.txt"; 
      defaultFile = true; 
   }
   System.out.println(configFile);
   Network network = new Network(configFile); 
  } //public static void main(String[] args) throws FileNotFoundException, IOException

   /*
   * Constructor for Network objects
   * Initializes the static class variable and reads the configuration file
   * Trains or runs the network depending on instructions from the config file
   * 
   * @throws FileNotFoundException throws FileNotFoundException if the selected config file can't be found 
   * @throws IOException if the output file that contains the weights from training cannot be written or overridden
   */
   public Network(String configFile) throws FileNotFoundException, IOException
   {
      time = 0.0;
      System.out.println();
      File config = new File(configFile); 
      Scanner scan = new Scanner(config); 

      String action = scan.nextLine();

      numRows = scan.nextInt(); 
      numHiddenLayers = scan.nextInt(); 
      numInputs = scan.nextInt(); 
      numLayers = numHiddenLayers + numHiddenLayers; 
      activationsPerHiddenLayer = new int[numHiddenLayers];
      for (int x = 0 ; x < numHiddenLayers ; x++)
      {
         activationsPerHiddenLayer[x] = scan.nextInt(); 
      } 
      numOutputs = scan.nextInt(); 

      lamda = scan.nextDouble();
      errorThreshold = scan.nextDouble();
      maxIterations = scan.nextInt(); 

      max = scan.nextDouble(); 
      min = scan.nextDouble(); 

      weightsFile = scan.next(); 
      outputFile = scan.next(); 

      boolean train;
      boolean random; 
      if (action.equals("train random"))
      {
         train = true;
         random = true; 
         System.out.println();
         System.out.println("-------------------------------------------------------------------"); 
         System.out.println("The network will be trained using random weights.");
         if (defaultFile)
         {
            System.out.println("Default config file used: " + configFile);
         }
         else
         {
            System.out.println("Config file used: " + configFile + " (from command line)");
         }
         System.out.print("Weights are saved to: " + outputFile); 
      } //if (action.equals("train random"))
      else if (action.equals("train preload"))
      {
         train = true; 
         random = false;
         System.out.println();
         System.out.println("-------------------------------------------------------------------"); 
         System.out.println("The network will be trained starting with pre-loaded weights from the weights file.");
         if (defaultFile)
         {
            System.out.println("Default config file used: " + configFile);
         }
         else
         {
            System.out.println("Config file used: " + configFile + " (from command line)");
         }
         System.out.print("Weights file used: " + weightsFile); 
         System.out.print("Weights are saved to: " + outputFile); 
      } //else if (action.equals("train preload"))
      else if (action.equals("run"))
      {
         train = false;
         random = false; 
         System.out.println();
         System.out.println("-------------------------------------------------------------------"); 
         System.out.println("The network will be run with the pre-loaded weights from the weights file.");
         if (defaultFile)
         {
            System.out.println("Default config file used: " + configFile);
         }
         else
         {
            System.out.println("Config file used: " + configFile + " (from command line)");
         }
         System.out.println("Weights file used: " + weightsFile); 
      } // else if (action.equals("run"))
      else
      {
         train = false; 
         random = false; 
         System.out.println();
         System.out.println("-------------------------------------------------------------------"); 
         System.out.print("Improper input detected in the first line of the config file.");
         System.out.println(" By default, the network will be run but not trained.");
         System.out.print("To specify the action, please enter either 'train random' to train using random weights, ");
         System.out.print("'train preload' to train using pre-loaded weights from the weights file, or 'run' to run the ");
         System.out.println("network in the first line of the config file."); 
      } //else 

      memoryAlloc(train);
      populate(scan, train, random); 

      if (train)
      {
         train();
      }
      else
      {
         run();
         System.out.println("-------------------------------------------------------------------"); 
      }
   } //public Network() throws FileNotFoundException, IOException 

   /*
   * Allocates memory for the program to store information about the network 
   * @param train true if the network is being trained, false otherwise 
   */
   public void memoryAlloc(boolean train)
   {
      network = new double[numLayers][];
      a = new double[numInputs];
      
      F = new double[numOutputs];

      mkWeights = new double[numInputs][activationsPerHiddenLayer[0]];
      kjWeights = new double[activationsPerHiddenLayer[0]][activationsPerHiddenLayer[1]]; 
      jiWeights = new double[activationsPerHiddenLayer[1]][numOutputs];

      if (train)
      {
         jTheta = new double[activationsPerHiddenLayer[1]];
         kTheta = new double[activationsPerHiddenLayer[0]]; 
         psiI = new double[numOutputs];
         psiJ = new double[activationsPerHiddenLayer[1]];
         psiK = new double[activationsPerHiddenLayer[0]]; 
      } // if (train)

      inputs = new double[numRows][]; 
      truth = new double[numRows][]; 

      for (int x = 1 ; x <= numHiddenLayers ; x++)
      {
         network[x] = new double[activationsPerHiddenLayer[x-1]]; 
      }

      for (int x = 0 ; x < numRows ; x++)
      {
         inputs[x] = new double[numInputs]; 
      }
   } //public void memoryAlloc(boolean train)

   /*
   * Populates the network's arrays with information about the network 
   * The information stored depends on whether the network is running, training with random weights, 
   * or training with preloaded weights.
   * 
   * If preloaded weights are used, then the weights file is read. 
   * This file contains the preloaded weights to use. The first line of the file contains the network configuration.
   * The second line of the file contains the mk weights. The third line of the file contains the kj weights. 
   * The fourth line of the file contaions the ji weights. Individual weights are separated by spaces. 
   * 
   * @param scan the scanner used to read the config file 
   * @param train true if the network is training, false otherwise 
   * @param random true if the network is being trained with random weights, false otherwise 
   * @throws FileNoteFoundException if the file cannot be found or read 
   */ 
   public void populate(Scanner scan, boolean train, boolean random) throws FileNotFoundException
   {
      File weights = new File(weightsFile); 
      Scanner scanWeights = new Scanner(weights);

      network[0] = a; 
      network[numLayers - 1] = F;

      scanWeights.nextLine();

      for (int x = 0 ; x < numRows ; x++)
      {
         for (int m = 0 ; m < numInputs ; m++)
         {
            inputs[x][m] = scan.nextDouble();
         }
      }

      for (int x = 0; x < numRows ; x++)
      {
         truth[x] = new double[numOutputs];
         for (int i = 0 ; i < numOutputs ; i++)
         {
            truth[x][i] = scan.nextDouble(); 
         }
      }

      for (int m = 0 ; m < numInputs ; m++)
      {
         for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)
         {
            if (train && random)
            {
               mkWeights[m][k] = randomNum(min,max);
            }
            else
            {
               mkWeights[m][k] = scanWeights.nextDouble();
            }
         } // for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)
      } // for (int m = 0 ; m < numInputs ; m++)

      for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)
      {
         for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)
         { 
            if (train && random)
            {
               kjWeights[k][j] = randomNum(min, max);
            }
            else
            {
               kjWeights[k][j] = scanWeights.nextDouble(); 
            }
         } // for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)
      } // for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)

      for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)
      {
         for (int i = 0 ; i < numOutputs ; i++)
         {
            if (train && random) 
            {
               jiWeights[j][i] = randomNum(min, max); 
            }
            else
            {
               jiWeights[j][i] = scanWeights.nextDouble(); 
            }
         } // for (int i = 0 ; i < numOutputs ; i++)
      } // for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)

      scanWeights.close(); 
      scan.close();
   } //public void populate(Scanner scan, boolean train, boolean random) throws FileNotFoundException
    
   /*
   * Implements the network for all the inputs
   * The network's result and the truth table are printed for each iteration
   * Each activation is connected to each activation in the next layer (i.e the network is feed-forward and fully connected) 
   */ 
   public void run()
   {
      for (int x = 0 ; x < numRows ; x++)
      {
         implementNetworkRun(x);
         if (x==0)
         {
            System.out.println("-------------------------------------------------------------------"); 
            System.out.println("Input, truth, and network output: ");
            System.out.println(); 
         }
         System.out.println(generateOutputString(inputs[x], truth[x], F)); 
      } //for (int x = 0 ; x < numRows ; x++)
   } //public void run()

   /*
   * Trains the network using backpropagation: 
   * Repeatedly calculates the partial derivative of the error function with respective to each weight, updating the weights 
   * accordingly to minimize the error with each iteration 
   * Iterates backwards from the last layer to avoid redundant calculations of intermediate terms in the chain rule
   * Saves the weights after training to the specified output file
   * 
   * @throws IOException throws an IOException if the output file containing the weights after training 
   *                     cannot be created or overriden
   */
   public void train() throws IOException
   {
      iterations = 0 ; 
      double start = System.currentTimeMillis();
      double end; 
      double[] omegaJ = new double[activationsPerHiddenLayer[1]]; 
      double omegaK; 

      do //while (error >= errorThreshold && iterations <= maxIterations)
      {
         for (int index = 0 ; index < numRows ; index++)
         {
            implementNetworkTrain(index);

            for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)
            {
               omegaJ[j] = 0.0; 
               for (int i = 0 ; i < numOutputs; i++)
               {
                  omegaJ[j] += psiI[i] * jiWeights[j][i]; 
                  jiWeights[j][i] += lamda * network[numHiddenLayers][j] * psiI[i];
               }
               psiJ[j] = omegaJ[j] * actFuncDeriv(jTheta[j]); 
            } //for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)

            for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)
            {
               omegaK = 0.0;
               for (int j = 0 ; j < activationsPerHiddenLayer[1]; j++)
               {
                  omegaK += psiJ[j] * kjWeights[k][j]; 
                  kjWeights[k][j] += lamda * network[numHiddenLayers-1][k] * psiJ[j];
               }
               psiK[k] = omegaK * actFuncDeriv(kTheta[k]); 

               for (int m = 0 ; m < numInputs ; m++)
               {
                  mkWeights[m][k] += lamda * a[m] * psiK[k]; 
               }
            } //for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)
         } //for (int index = 0 ; index < numRows ; index++)
         iterations++; 
         error = calcError(); 
      } while (error >= errorThreshold && iterations <= maxIterations); 

      saveWeights(); 
      end = System.currentTimeMillis(); 
      time = end - start; 
      printNetworkInfo(); 
   } // public void train() throws IOException

   /*
   * Implements a single input into the network as a part of running the network. 
   * Stores the input activations and combines them with other input activations according to the weights 
   * in order to populate the hidden layer. The hidden layer values are then combined, again according to the weights, 
   * in order to populate the output layer. 
   * 
   * @param index the index of the input that will be implemented into the network 
   * @param train true if the network being trained, false otherwise 
   */
   private void implementNetworkRun(int index)
   {
      double thetaI; 
      double thetaJ; 
      double thetaK; 
      double[] mkNodes; 
      double[] kjNodes;

      a = inputs[index];
      mkNodes = network[1];
      kjNodes = network[2]; 

      for (int i = 0 ; i < numOutputs ; i++)
      {
         thetaI = 0.0; 
         for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)
         {
            thetaJ = 0.0; 
            for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)
            { 
               thetaK = 0.0; 
               for (int m = 0 ; m < numInputs ; m++)
               {
                  thetaK += a[m] * mkWeights[m][k]; 
               }
               mkNodes[k] = actFunc(thetaK); 
               thetaJ += mkNodes[k] * kjWeights[k][j]; 
            } //for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)

            kjNodes[j] = actFunc(thetaJ); 
            thetaI += kjNodes[j] * jiWeights[j][i]; 
         } //for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)

         F[i] = actFunc(thetaI); 
      } // for (int i = 0 ; i < numOutputs ; i++)
   } //private void implementNetwork(int index, boolean train) 

   /*
   * Implements a single input into the network as a part of training the network.
   * The difference between this method and implementNetworkRun() is that here, values such as kTheta, which are only 
   * needed in training and not running, are updated. 
   * Stores the input activations and combines them with other input activations according to the weights 
   * in order to populate the hidden layer. The hidden layer values are then combined, again according to the weights, 
   * in order to populate the output layer. 
   * 
   * @param index the index of the input that will be implemented into the network 
   * @param train true if the network being trained, false otherwise 
   */
   private void implementNetworkTrain(int index)
   {
      double thetaI; 
      double thetaJ; 
      double thetaK; 
      double[] mkNodes; 
      double[] kjNodes;

      a = inputs[index];
      mkNodes = network[1];
      kjNodes = network[2]; 

      for (int i = 0 ; i < numOutputs ; i++)
      {
         thetaI = 0.0; 
         for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)
         {
            thetaJ = 0.0; 
            for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)
            { 
               thetaK = 0.0; 
               for (int m = 0 ; m < numInputs ; m++)
               {
                  thetaK += a[m] * mkWeights[m][k]; 
               }
               mkNodes[k] = actFunc(thetaK); 
               thetaJ += mkNodes[k] * kjWeights[k][j]; 
               kTheta[k] = thetaK; 
              
           } //for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)

            kjNodes[j] = actFunc(thetaJ); 
            thetaI += kjNodes[j] * jiWeights[j][i]; 
            jTheta[j] = thetaJ; 
           
         } //for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)

         F[i] = actFunc(thetaI); 
         double omegaI = truth[index][i] - F[i]; 
         psiI[i] = omegaI * actFuncDeriv(thetaI); 
     } // for (int i = 0 ; i < numOutputs ; i++)
  } //private void implementNetwork(int index, boolean train) 

   /*
   * Calculates the error of the network:
   * Compares the output to the truth table output for each row of inputs
   * The error is calculated with the following formula: Error = 0.5 * Sum(Ti - Fi)^2
   * In words, the error is one half of the sum of the individual errors, where an individual error for a given input 
   * row is the square of the difference between the theoretical output and the network's output.
   * 
   * @return the error value
   */
   private double calcError()
   {
      error = 0.0; 
      for (int index = 0 ; index < numRows ; index++)
      {
         implementNetworkTrain(index); 
         for (int i = 0 ; i < numOutputs ; i++)
         {
            error += (truth[index][i] - F[i]) * (truth[index][i] - F[i]);
         } 
      }
      return 0.5 * error; 
   } //private double calcError() 

   /*
   * The network's activation function, which is applied to the activation state of each node 
   * In this network, the activation function is the sigmoid function.
   * 
   * @param theta The activation state that the function will be applied to 
   * @return The value of activation function at the given parameter value 
   */
   private double actFunc(double theta)
   {
      double res = (1.0 / (1.0 + Math.exp(-theta)));
      return res;
   }
  
   /*
   * The derivative of the network's activation function
   * As written, this method uses the specific activation function of this network, which is the sigmoid function.
   * 
   * @param theta The activation state that the function will be applied to
   * @return The value of the derivative of the activation function at the given parameter value 
   */
   private double actFuncDeriv(double theta)
   {
      double aTheta = actFunc(theta); 
      double res = aTheta * (1.0 - aTheta); 
      return res; 
   }
  
   /*
   * Generates a random number (double) within a specified range 
   * 
   * @param low The lower bound of the range 
   * @param high The upper bound of the range
   * 
   * @return A random double within a given range 
   */
   private double randomNum(double low, double high)
   {
      double range = high - low;
      return ((Math.random() * range) + low); 
   }

   /*
   * Generates a string with some basic information about the values being passed through the network
   * The information includes the input values, the truth table values, and the network's output values at a specified index.
   * 
   * @param input An array of the current inputs 
   * @param truth The truth table output at the specified index 
   * @param out The current output at the specified index 
   * 
   * @return A string containing the input, the truth table, and the network's output. 
   */
   private String generateOutputString(double[] input, double[] truth, double[] out)
   {
      String str = ("Truth: (");
      for (double i : truth)
      {
         str += String.valueOf(i) + ", "; 
      }
      str = str.substring(0, str.length() - 2);

      str += ") ; Output: (";
      for (double i : out) 
      {
         str += String.valueOf(i) + ", ";
      }
      str = str.substring(0, str.length() - 2);
      str += ")";

      return str;
   } //private String generateOutputString(double[] input, double[] truth, double[] out)

   /*
   * Saves the weights from training into an output file
   * The first line of the file will include the configuration of the network. 
   * The next line of the file contains the mk weights, separated by spaces. 
   * The next line of the file contains the kj weights, separated by spaces. 
   * The next line of the file contains the ji weights, separated by spaces. 
   * @throws IOException if the file "output.txt" cannot be written or overridden 
   */
   private void saveWeights() throws IOException
   {
      File output = new File(outputFile);
      FileWriter fileWriter = new FileWriter(output);

      fileWriter.write("Network configuration: " + numInputs + "--" + activationsPerHiddenLayer[0]);
      fileWriter.write( "--" + activationsPerHiddenLayer[1] + "--" + numOutputs);
      fileWriter.write("\n");

      for (int m = 0 ; m < numInputs ; m++)
      {
         for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)
         {
            fileWriter.write(mkWeights[m][k] + " "); 
         }
      }

      fileWriter.write("\n"); 
      for (int k = 0 ; k < activationsPerHiddenLayer[0] ; k++)
      {
         for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)
         {
            fileWriter.write(kjWeights[k][j] + " "); 
         }
      }

      fileWriter.write("\n");
      
      for (int j = 0 ; j < activationsPerHiddenLayer[1] ; j++)
      {
         for (int i = 0 ; i < numOutputs ; i++)
         {
            fileWriter.write(jiWeights[j][i] + " "); 
         }
      } 

      fileWriter.close(); 
      
   System.out.println();
   } //private void saveWeights() throws IOException

   /*
   * Prints the entire information of the network 
   * 
   * When running, the method prints the input, the truth table, and the output. 
   * 
   * When training, the method prints the network configuration and the 
   * reason the network stopped training (i.e. the error is lower than the threshold or the 
   * max number of iterations has been reached).
   * The error is also printed. If the error is lower than the threshold, the number of iterations required is also printed. 
   * The method also prints the lamda value used.
   * The method also creates or overrides an output file that contains the value of the weights after training. 
   * 
   * @throws IOException Throws an IOException if the output file cannot be written to or created. 
   */
   private void printNetworkInfo() throws IOException
   {
      System.out.println("-------------------------------------------------------------------"); 
      System.out.println("Network Information: ");
      System.out.println(); 
      System.out.println("Number of rows in truth table: " + numRows);
      System.out.println("Number of input nodes: " + numInputs);
      System.out.println("Activations in first hidden layer: " + activationsPerHiddenLayer[0]);
      System.out.println("Activations in second hidden layer: " + activationsPerHiddenLayer[1]);
      System.out.println("Number of outputs: " + numOutputs);
      System.out.print("Network configuration: " + numInputs + "--" + activationsPerHiddenLayer[0]);
      System.out.println("--" + activationsPerHiddenLayer[1] + "--" + numOutputs);
      System.out.println(); 

      System.out.println("Training Information: ");
      System.out.println(); 

      System.out.println("Training time (ms): " + time);
      if (error < errorThreshold)
      {
         System.out.println("Reason for exit: error is less than the error threshold of " + errorThreshold);
         System.out.println("Error: " + error);
         System.out.println("Number of iterations required: " + iterations);
         System.out.println("Max number of iterations allowed: " + maxIterations);
      }
      else 
      {
         System.out.println("Reason for exit: max iterations of " + maxIterations + " reached");
         System.out.println("Error: " + error);
         System.out.println("Error threshold used: " + errorThreshold);
      }

      System.out.println("Lamda value used: " + lamda);
      System.out.println("Number range for random number generator: " + min + " - " + max);
      
      run();
      saveWeights(); 
      System.out.println("-------------------------------------------------------------------"); 
 } //private void printNetworkInfo(boolean print) throws IOException 
} //public class Network 
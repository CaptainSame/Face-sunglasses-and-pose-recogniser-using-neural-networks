/*
X = Mx960(then add 1 column of ones)
input_layer_size  = 960; //32x30 images 
hidden_layer_size_face = 20; theta1 = 20x961 
hidden_layer_size_pose = 6; theta1 = 6x961
learning_rate = 0.3
num_labels_sunglasses = 2; theta2 = , Y = Mx2
num_labels_face = 20; theta2 = 20x21, Y = Mx20
num_labels_pose = 4; theta2 = 4x7, Y = Mx4
*/
import java.util.Random;
import java.io.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

class Glass{
	
	static int n1=960,n2=550,n3=2;
	static double alpha = 0.011, momentum = 0.001;
	
	
	public static double[] sigmoid(double z[], int size){
		int i, j;
		double [] sig = new double[size];
		for(i=0; i<size; i++){
			sig[i] = 1/(1 + Math.exp(-1*z[i]));
		}
		return sig;
	}
	
	public static double[] sig_gradient(double z[], int size){
		int i, j;
		double [] grad = new double[size];
		for(i=0; i<size; i++){	
			grad[i] = z[i]*(1 - z[i]);
		}
		return grad;
	}
	
	static double[][] random_initialize(int x, int y){
		int i, j;
		Random rand = new Random();
		double [][] initial_array = new double[x][y];
		for(i=0; i<x; i++){
			for(j=0; j<y; j++){
				initial_array[i][j] = rand.nextDouble()*0.3 - 0.15;
			}
		}
		return initial_array;		
	}
	
	public static double[] create_labels(String s, int n3){
		
		int id=0;
		double [] y = new double[n3];
		for(int i=0;i<n3;i++) y[i]=0;
		
		String parts[] = s.split("/");
		String attributes[] = parts[2].split("_");
		
		String t = attributes[3];
		if(t.equals("open")) id = 0;
		else if(t.equals("sunglasses")) id = 1;
				
		y[id] = 1;
		
		return y;
	}
	
	
	public static void main(String args[]){
		
	//Neural n = new Neural();
	int i,j,k,l = 200;
	double sum=0;
	double [][] theta1,theta2, theta1_grad, theta2_grad, prev_theta1_grad, prev_theta2_grad;
	double [] X, a2, a3, s2, s3, delta2, delta3, Y, temp;
	
	theta1_grad=new double[n2][n1+1];
	theta2_grad=new double[n3][n2+1];
	prev_theta1_grad=new double[n2][n1+1];
	prev_theta2_grad=new double[n3][n2+1];
	theta1=random_initialize(n2,n1+1);
	theta2=random_initialize(n3,n2+1);
	
	for(i=0;i<n2;i++){
		//System.out.println("theta1");
		for(j=0;j<n1+1;j++){
			prev_theta1_grad[i][j] = 0;;
		}
	}
	
	for(i=0;i<n3;i++){
		//System.out.println("theta2");
		for(j=0;j<n2+1;j++){
			prev_theta2_grad[i][j] = 0;
		}
	} 
	
	X = new double[n1+1];
	a2 = new double[n2];
	a3 = new double[n3];
	s2 = new double[n2+1];
	temp = new double[n2+1];
	s3 = new double[n3];
	delta3 = new double[n3];
	delta2 = new double[n2];
	Y = new double[n3]; // or int??
	
	
	String trainFile="glasses_train.list";
	FileInputStream fin;
	
	try {
	
	while(l-- >= 0){
		
		//System.out.println("");
		
	//System.out.println(l);
	BufferedReader br = new BufferedReader(new FileReader(trainFile));
	
    String line;
    while ((line = br.readLine()) != null) {
//////////////////////////////////////////////////////////////////		
		
		try{
			fin = new FileInputStream(line);
		}
		catch(FileNotFoundException e){
			System.out.println("file open error");
			return;
		}
		
		i=0;
		j=0;
		k=1;
		X[0]=1;
		try{
		while(i!=-1){
			i=fin.read();
			if(k>n1) break;
			if(i==10)j++;
			if(j<3){
					//System.out.print((char)i);
					
				}
				//else if(j==3)System.out.print((char)i);
			else{
				
				if (i!=-1||i!=32||i!=10){
					//System.out.print(i);
					//System.out.print(" ");
					X[k]=i;
					k++;
					}
				}
		}
		
		}
		catch(IOException e){
			System.out.println("file read error");
			return;
		}
		
		try{
			fin.close();
		}
		catch(IOException e){
			System.out.println("file close error");
		}
		
		
///////////////////////////////////////////////////////////////////////
	/*training using stochastic gradient descent i.e., one example at a time. */
	
	
		Y = create_labels(line, n3);
		
		//System.out.println("list file opened. " + line);
		for(j=0;j<n2;j++){
				sum = 0;
			for(k=0;k<n1+1;k++){
				sum = sum + X[k]*theta1[j][k];
			}
			a2[j] = sum; 
		}
		
		a2 = sigmoid(a2, n2);
		
		s2[0] = 1;
		for(i=1;i<n2+1;i++){
			s2[i] = a2[i-1];
		}
		
		for(j=0;j<n3;j++){
				sum = 0;
			for(k=0;k<n2+1;k++){
				sum = sum + s2[k]*theta2[j][k];
			}
			a3[j] = sum; 
		}
		
			
		a3 = sigmoid(a3, n3);
		/*
		System.out.println("final");
			
			for(i=0;i<n3;i++){
				System.out.print(" " + a3[i]);
			}
		*/
		s3 = a3;
			
		for(i=0;i<n3;i++){
			delta3[i] = s3[i] - Y[i];
		}
		/* ********************* */
		a3 = sig_gradient(a3, n3);
		
		for(i=0;i<n3;i++){
			delta3[i] = delta3[i]*a3[i];
		}
		/* ******************* */
		
		for(i=0;i<n3;i++){
			for(j=0;j<n2+1;j++){
				theta2_grad[i][j] = delta3[i]*s2[j];
			}
		}
		
		for(i=0;i<n2+1;i++){
			sum=0;
			for(j=0;j<n3;j++){
				sum = sum + delta3[j]*theta2[j][i];
			}
			temp[i] = sum;
		}
		
		a2 = sig_gradient(a2, n2);
		
		for(i=0;i<n2;i++){
			delta2[i] = temp[i+1]*a2[i];
		}
		
		for(i=0;i<n2;i++){
			for(j=0;j<n1+1;j++){
				theta1_grad[i][j] = delta2[i]*X[j];
			}
		}
		
		for(i=0;i<n2;i++){
			for(j=0;j<n1+1;j++){
				theta1_grad[i][j] = alpha*theta1_grad[i][j] + momentum*prev_theta1_grad[i][j];
				theta1[i][j] = theta1[i][j] - theta1_grad[i][j];
			}
		}
		
		for(i=0;i<n3;i++){
			for(j=0;j<n2+1;j++){
				theta2_grad[i][j] = alpha*theta2_grad[i][j] + momentum*prev_theta2_grad[i][j];
				theta2[i][j] = theta2[i][j] - theta2_grad[i][j];
			}
		}
		
		
		for(i=0;i<n2;i++){
			//System.out.println("theta1");
			for(j=0;j<n1+1;j++){
				prev_theta1_grad[i][j] = theta1_grad[i][j];
			}
		}
		
		for(i=0;i<n3;i++){
			//System.out.println("theta2");
			for(j=0;j<n2+1;j++){
				prev_theta2_grad[i][j] = theta2_grad[i][j];
			}
		}
		
	}  
	} //while bracket 
	} //try bracket
	catch(Exception e){
		System.out.println("first catch");
	}
	/*
		for(i=0;i<n2;i++){
			System.out.println("theta1");
			for(j=0;j<n1+1;j++){
				System.out.print(" " + theta1[i][j]);
			}
		}
		
		for(i=0;i<n3;i++){
			System.out.println("theta2");
			for(j=0;j<n2+1;j++){
				System.out.print(" " + theta2[i][j]);
			}
		}
		System.out.println("end");
	*/
	/* training complete */
	int dobaar=2;
	while(dobaar>0){
		dobaar--;
		if(dobaar==1)trainFile="glasses_test1.list";
		else if(dobaar==0)trainFile="glasses_test2.list";
		/* testing examples one at a time */
		////////////////////////////////////////////////////////////////
		try (BufferedReader br = new BufferedReader(new FileReader(trainFile))) {
		String line;
		
		double correct = 0, incorrect = 0;
		while ((line = br.readLine()) != null) {
	//////////////////////////////////////////////////////////////////		
			try{
				fin = new FileInputStream(line);
			}
			catch(FileNotFoundException e){
				System.out.println("file open error");
				return;
			}
			i=0;
			j=0;
			k=1;
			X[0]=1;
			try{
			while(i!=-1){
				
				if(k>n1) break;
				i=fin.read();
				if(i==10)j++;
				if(j<3){
						//System.out.print((char)i);
						
					}
					//else if(j==3)System.out.print((char)i);
				else{
					
					if (i!=-1||i!=32||i!=10){
						//System.out.print(i);
						//System.out.print(" ");
						X[k]=i;
						k++;
						}
					}
			}
			}
			catch(IOException e){
				System.out.println("file read error");
				return;
			}
			/*
			System.out.println("test_inputs");
			for(i=0;i<n1;i++) System.out.print(" " + X[i]);
			System.out.println("\n\n");
			*/
			try{
				fin.close();
			}
			catch(IOException e){
				System.out.println("file close error");
			}
		
		///////////////////////////////////////////////////////////////
			Y = create_labels(line, n3);
			
			for(j=0;j<n2;j++){
					sum = 0;
				for(k=0;k<n1+1;k++){
					sum = sum + X[k]*theta1[j][k];
				}
				a2[j] = sum; 
			}
			
			a2 = sigmoid(a2, n2);   
			
			s2[0] = 1;
			for(i=1;i<n2+1;i++){
				s2[i] = a2[i-1];
			}
			
			for(j=0;j<n3;j++){
					sum = 0;
				for(k=0;k<n2+1;k++){
					sum = sum + s2[k]*theta2[j][k];
				}
				a3[j] = sum; 
			}
			
			a3 = sigmoid(a3, n3);
			s3 = a3;
			/*System.out.println("");
			System.out.println("output");
			for(i=0;i<n3;i++){
				System.out.print(" " + s3[i]);
				System.out.println(" " + Y[i]);
			}
			*/
			
			double max=0;
			int maxindex = 0, flag = 0;
			
			for(i=0;i<n3;i++){
				if(s3[i]>max){
					max = s3[i];
					maxindex = i;
				}
			}
			
			if(Y[maxindex] == 1) correct = correct + 1;
			else incorrect = incorrect + 1;
			/*
			for(i=0;i<n3;i++){
				//System.out.println();
				if(!(((s3[i]<=0.5)&&(Y[i]==0)) || ((s3[i]>0.5)&&(Y[i]==1)))) {
					flag = 1;
					break;
				}
			}
			
			if(flag==1) incorrect = incorrect + 1;
			else correct = correct + 1;
			*/
		}  //list bracket
		System.out.println("");
		double accuracy = correct/(correct+incorrect);
		System.out.println(accuracy*100);
		
		} //try bracket
		
		catch(Exception e){
			System.out.println("second catch" + dobaar);
		}
	}  //2 loop bracket
	
	}  //mains bracket
}
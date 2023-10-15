#include <armadillo>
#include <iostream>
#include <cmath>

using namespace arma; 

class NN_1Layer{
public:
    NN_1Layer(int nodes, double bias): nodes(nodes), bias(bias), 
                weights(2, nodes, fill::randn){};
    int nodes;
    double bias;
    mat weights;

    double forward_dep(double input){
        /**
         * deprecated
        */
        double z1 = this->weights(0, 0) * input + this->bias;
        double z2 = this->weights(0, 1) * input + this->bias;

        double a1 = this->activation(z1);
        double a2 = this->activation(z2);

        double out = this->weights(1,0)*a1 + this->weights(1,1)*a2;
        return out;
    }

    double forward(const double input){
        /**
         * array style forward function
        */
        auto w0 = this->weights.row(0);
        auto w1 = this->weights.row(1);
        auto z = input * w0 + this->bias;

        double out = dot(z,w1);
        return out;
    }

    double activation(double input){
        return input;
    }

    void gdStep(double learning_rate, double loss, double x){
        mat dw0, dw1;
        double db;

        dw0 = this->weights.row(1) * x;
        dw1 = this->weights.row(0) * x + this->bias;
        db = accu(this->weights.row(1));

        this->bias -= learning_rate * db;

        this->weights.row(0) -= learning_rate * dw0 * loss;
        this->weights.row(1) -= learning_rate * dw1 * loss;

    }

    void gdStep_old(double learning_rate, double loss, double x){
        double dw00, dw01, dw10, dw11, db;

        dw00 = this->weights(1, 0) * x;
        dw01 = this->weights(1, 1) * x;
        dw10 = this->weights(0, 0) * x + this->bias;
        dw11 = this->weights(0, 1) * x + this->bias;
        db = this->weights(1, 0) + this->weights(1, 1);

        this->weights(0, 0) -= learning_rate * dw00 * loss;
        this->weights(1, 0) -= learning_rate * dw10 * loss;
        this->weights(0, 1) -= learning_rate * dw01 * loss;
        this->weights(1, 1) -= learning_rate * dw11 * loss;

        this->bias -= learning_rate * db;

    }

    void print_data(){
        mat out(100, 2, fill::zeros);
        out.col(0) = linspace(-1, 1, 100);
        for (int i=0; i<100; ++i){
            out(i, 1) = this->forward(out(i, 0));
        }
        std::cout << "got here" << std::endl;
        out.save("testdata.txt", raw_ascii);
    }
};

class NeuralNet{
    /**
     * this will be a fully connected network (?)
     * should we leave the number of layers and nodes up?
     * no, for the first one fix them
    */
public:
    NeuralNet(int layers, int nodes): layers(layers), nodes(nodes), 
              weights(layers, nodes, fill::randn), biases(layers, fill::zeros) {};

    int layers, nodes;
    mat weights;
    vec biases;

    double forward(double input){
        for (int i=0; i<this->layers; ++i){
            i++;
        }
        return 0.0;
    }



private:
    double activation(double input){
        /**
         * let's start by doing no activation?
         * Leaving this function in so that we can write it as if we have
         * a non-identity activation function.
        */
        return input;
    }
};

double f(double x){
    return x*x;
}

double loss( double yhat, double y ){
    return 0.5 * (yhat - y)*(yhat - y);
}

mat f( mat x ){
    return x%x;
}

mat genData(int count){
    mat out(count, 2, fill::zeros);
    out.col(0) = randu(count, distr_param(-2, 2));
    out.col(1) = f(out.col(0));
    return out;
}

int main(){
    /**
     * for this test let's interpolate f(x) = x^3
    */


   NN_1Layer nn(3, 0.0);

   int dataLen = 2500;
   auto tdata2(genData(dataLen));

   double output, li, tol;
   tol = 1.e-6;
   int max_iter = 1000000;
   double learning_rate = 0.0001;
   double x, fx, running_loss(0), last_loss(0);
   int double_idx(1);
   for (int idx=0; idx<max_iter; ++idx){
    for (int dataIdx=0; dataIdx<dataLen; ++dataIdx){
        auto data = tdata2.row(dataIdx);

        x = data(0);
        fx = data(1);

        output = nn.forward(x);
        li = loss( output, fx );
        // std::cout << "x: " << x << std::endl; 
        // std::cout << "fx: " << fx << std::endl; 
        // std::cout << "output: " << output << std::endl; 
        // std::cout << "loss: " << li << std::endl; 
        nn.gdStep(learning_rate, output - fx , x);

        running_loss += li;
    }

    if( running_loss / (100 * double_idx) < tol ){
        std::cout << "Hit tolerance after " << idx << " steps." << std::endl;
        std::cout << "Final loss: " << running_loss << std::endl; 
        std::cout << "Final weight matrix: " << std::endl;
        nn.weights.print();

        nn.print_data();
        return 0;
    }

    if( idx % 100 == 0){
        if( idx > 4500 && abs(running_loss/(100*dataLen) - last_loss) < tol){
            std::cout << "Stopped converging" << std::endl;
            std::cout << "Final loss: " << running_loss/(100*dataLen) << std::endl;
            std::cout << "stopping condn: " << running_loss/(100*dataLen) - last_loss << std::endl;
            std::cout << "final weights: " << std::endl;
            nn.weights.print();
            nn.print_data();
            return 0;
        }
        std::cout << "Finished minibatch " << idx << std::endl;
        std::cout << "Running loss: " << running_loss / (100 * dataLen) << std::endl;
        nn.weights.print();
        last_loss = running_loss / (100 * dataLen);
        running_loss = 0;
        double_idx = 1;
    }
   }

//    std::cout << nn.forward_dep(3.0) << std::endl;

//    

//    tdata2.print();
   return 0;
}
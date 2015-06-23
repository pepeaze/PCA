#include <stdlib.h>
#include <string>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <fstream>
#include <sstream>
#include <string.h>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp> 
#include <boost/lambda/bind.hpp>
#include <Eigen/Dense> /*Eigen and boost is mostly header-only library. All that you need is to add Eigen path to (MSVC2010):
					 Project Properties -> C/C++ -> General -> Additional Include Directories
					 Let's say you have header Core in folder C:/folder1/folder2/Eigen/, i.e.
					 C:/folder1/folder2/Eigen/Core
					 So you should add path C:/folder1/folder2 to Additional Include Directories.*/
#include <Eigen/SVD>

using namespace std;
namespace fs = boost::filesystem;
//using namespace boost::filesystem;
using namespace boost::lambda;
using namespace Eigen;

clock_t inicio, fim;

struct vertices{

	float pX;
	float pY;
	float pZ;
};

vertices vert;
vector<vertices> v;

float **cria_matriz(int linhas, int colunas){

    float **matriz;
    int i, j;

    matriz = (float**)malloc(linhas * sizeof(float*)); //Aloca um Vetor de Ponteiros

    for (i = 0; i < linhas; i++){ //Percorre as linhas do Vetor de Ponteiros
       matriz[i] = (float*) malloc(colunas * sizeof(float)); //Aloca um Vetor de Inteiros para cada posição do Vetor de Ponteiros.
       for (j = 0; j < colunas; j++){ //Percorre o Vetor de Inteiros atual.
            matriz[i][j] = 0; //Inicializa com 0.
       }
	}


    if (!matriz){
        printf("Falta memoria para alocar o vetor de ponteiros");
        exit(1);
    }

    return matriz;

}

void parseOBJ(string arq){

	FILE * file = fopen(arq.c_str(), "r");
	if( file == NULL ){
		printf("Nao consegui abrir o arquivo!\n");
	}

	while( 1 ){
 
		char lineHeader[128];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF)
			break; //
		
		if ( strcmp( lineHeader, "v" ) == 0 ){
			fscanf(file, "%f %f %f\n", &vert.pX, &vert.pY, &vert.pZ);
			v.push_back(vert);
		}

	}

	fclose(file);

}

/*a biblioteca Boost possui um bom gerenciador de diretorios. Com ele, sou capaz de
percorrer o diretorio e pegar as informacoes iniciais que preciso*/
void getFolderInf(string nomeFolder, string &nomePrimArq, int &numTotalArquivos){

	fs::path targetDir(nomeFolder); 

	fs::directory_iterator it(targetDir), eod;
	numTotalArquivos = 0;
	int pegueiNome=0;

	BOOST_FOREACH(fs::path const &p, std::make_pair(it, eod))//percorre todo o diretorio   
	{ 
		if(is_regular_file(p))
		{
			if(pegueiNome==0){
				stringstream ss;
				ss << p;
				nomePrimArq = ss.str();
				pegueiNome = 1;
			}
			numTotalArquivos++;
		} 
	}
}

void preencheListV(int linhas, int colunas, MatrixXf &listV, string nomeFolder){
	inicio = clock();
	cout<<"Preenchendo listV"<<endl;
	int file = 0;
	fs::path targetDir(nomeFolder); 
	fs::directory_iterator it(targetDir), eod;
	string nomeArquivo;
	string nomeArquivoWQ;

	BOOST_FOREACH(fs::path const &p, std::make_pair(it, eod))   
	{ 
		if(is_regular_file(p))
		{
			stringstream ss;
			ss << p;
			nomeArquivo = ss.str();
			nomeArquivoWQ = nomeArquivo.substr(1,(nomeArquivo.size()-2)); //acerto a string com o nome do arquivo. A mesma retorna da biblioteca boost com ""
			parseOBJ(nomeArquivoWQ); //faço o parse do arquivo obj e preencho o vetor de vertices do arquivo atual no loop
			
			//preencho listV na linha "file" com os valores do vetor de vertices
			for(int j=0; j<(colunas/3); j++){
				listV(file,j) = v[j].pX;
				listV(file,j+v.size()) = v[j].pY;
				listV(file,j+(v.size()*2)) = v[j].pZ;
				/*listV[file][j] = v[j].pX;
				listV[file][j+v.size()] = v[j].pY;
				listV[file][j+(v.size()*2)] = v[j].pZ;*/							
			} 
			
			v.clear(); //limpo o vetor de vertices para começar uma nova iteração
			//cout<<"Passei pelo arquivo "<<file+1<<"  "<<listV(file,0)<<endl;
			file++; //incremento a linha da matriz
		}
	}
	fim = clock();
	printf("Levei: %lf segundos\n\n",((double)(fim - inicio)/CLOCKS_PER_SEC));
	//cout<<listV(0,0)<<endl;

}

void calcMedia(int linhas, int colunas, MatrixXf models, MatrixXf &avg){
	inicio = clock();
	cout<<"Calculando media"<<endl;
	int l, c;
	l = colunas;
	c = linhas;
	float media = 0;
	float sumRow = 0;

	for(int i = 0; i< l; i++){
		for(int j = 0; j< c; j++){
			sumRow+=models(i,j);
			media = sumRow/colunas;
					
		}
		avg(i,0) = media;
		media = 0;
		sumRow = 0;
		//cout<<avg(i,0)<<endl;
	}
	fim = clock();
	printf("Levei: %lf segundos\n\n",((double)(fim - inicio)/CLOCKS_PER_SEC));
}

void subtraiDimensaoMedia(int linhas, int colunas, MatrixXf &models, MatrixXf avg){
	inicio = clock();
	cout<<"Subtraindo dimensao da media"<<endl;
	int l, c;
	l = colunas;
	c = linhas;

	for(int i = 0; i< l; i++){
		for(int j = 0; j< c; j++){
			models(i,j) -= avg(i,0);		
		}
	}
	fim = clock();
	printf("Levei: %lf segundos\n\n",((double)(fim - inicio)/CLOCKS_PER_SEC));

}

JacobiSVD<MatrixXf> calcSVD(MatrixXf models){
	inicio = clock();
	cout<<"Calculando SVD"<<endl;
	JacobiSVD<MatrixXf> svd(models, ComputeThinU | ComputeThinV);
	fim = clock();
	printf("Levei: %lf segundos\n\n",((double)(fim - inicio)/CLOCKS_PER_SEC));
	return svd;
}

int main(){

	string nomeFolder = "D:\\MestradoUFES\\projetoMestrado\\FacesMorfologicas\\male10K_1";
	string nomePrimArq, nomePrimArqWQ; //preciso do nome do primeiro arquivo para descobrir o tamanho de  linhas "v" e alocar na matriz listV como colunas
	int numTotalArquivos; //numero total de linhas da matriz listV
	//float **listV;//matriz dos vertices/arquivo
	int linhas, colunas;
	MatrixXf listV(1,1);
	MatrixXf models(1,1);
	MatrixXf avg(1,1);
	JacobiSVD<MatrixXf> svd;

	getFolderInf(nomeFolder, nomePrimArq, numTotalArquivos); //pego as informacoes iniciais para alocar listV
	//cout<<nomePrimArq<<"  "<<numTotalArquivos<<endl;
	nomePrimArqWQ = nomePrimArq.substr(1,(nomePrimArq.size()-2)); //acerto a string com o nome do arquivo. A mesma retorna da biblioteca boost com ""
	parseOBJ(nomePrimArqWQ); //faço o parse do arquivo obj e preencho o vetor de vertices do primeiro arquivo
	linhas = numTotalArquivos;
	colunas = (v.size()*3);
	listV.resize(linhas,colunas);
	models.resize(colunas, linhas);
	avg.resize(colunas,1);
	v.clear();
	//listV = cria_matriz(linhas, colunas); //aloco a lista de vertices
	preencheListV(linhas, colunas, listV, nomeFolder); //preencho a lista de vertices
	models = listV.transpose();
	//cout<<models(0,0)<<endl;
	calcMedia(linhas,colunas, models, avg); //calcula modelo médio
	subtraiDimensaoMedia(linhas, colunas, models, avg);
	//cout<<models(0,0)<<endl;
	//calcSVD(models);
	svd = calcSVD(models);
	//cout<<svd.singularValues()<<endl;

	cin.get();
}
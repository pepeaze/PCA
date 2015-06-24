/*COISAS OPENGL*/
#include <Windows.h>//obrigatório incluir para usar o opengl
#include <gl/GL.h>
#include <gl/GLU.h>
#include <gl/glut.h>
/****************/

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

/*altura e largura da janela 
de visualização do modelo*/
int windowW = 640;
int windowH = 480;
/*************************/

/*COISAS PCA*/
int linhas, colunas;
MatrixXf listV(1,1);
MatrixXf models(1,1);
MatrixXf avg(1,1);
JacobiSVD<MatrixXf> svd;
/************/
clock_t inicio, fim;

struct vertices{

	float pX;
	float pY;
	float pZ;
};

struct faces{

	float pX;
	float pY;
	float pZ;
};

vertices vert;
faces fac;

vector<vertices> v;
vector<faces> f;

void salvaAVG(MatrixXf avg){

	ofstream arq;
	arq.open("avg.txt");
	for(int i =0; i<avg.rows();i++){
		arq<<avg(i,0)<<endl;
	}
	arq.close();

}

void parseOBJ(string arq, int preencheF){

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

		if (preencheF == 1){
		
			if ( strcmp( lineHeader, "f" ) == 0 ){
				fscanf(file, "%f %f %f\n", &fac.pX, &fac.pY, &fac.pZ);
				f.push_back(fac);
			}
		
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
			parseOBJ(nomeArquivoWQ, 0); //faço o parse do arquivo obj e preencho o vetor de vertices do arquivo atual no loop
			
			//preencho listV na linha "file" com os valores do vetor de vertices
			for(int j=0; j<(colunas/3); j++){
				listV(file,j) = v[j].pX;
				listV(file,j+v.size()) = v[j].pY;
				listV(file,j+(v.size()*2)) = v[j].pZ;							
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

void PCA(){

	string nomeFolder = "D:\\MestradoUFES\\projetoMestrado\\FacesMorfologicas\\testFolder";
	string nomePrimArq, nomePrimArqWQ; //preciso do nome do primeiro arquivo para descobrir o tamanho de  linhas "v" e alocar na matriz listV como colunas
	int numTotalArquivos; //numero total de linhas da matriz listV
	int preencheF = 1;
	getFolderInf(nomeFolder, nomePrimArq, numTotalArquivos); //pego as informacoes iniciais para alocar listV
	//cout<<nomePrimArq<<"  "<<numTotalArquivos<<endl;
	nomePrimArqWQ = nomePrimArq.substr(1,(nomePrimArq.size()-2)); //acerto a string com o nome do arquivo. A mesma retorna da biblioteca boost com ""
	parseOBJ(nomePrimArqWQ, preencheF); //faço o parse do arquivo obj, preenchendo o vetor de vertices do primeiro arquivo e o vetor de faces uma unica vez pois o mesmo se repete em todos os arquivos
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
	//salvaAVG(avg);
	subtraiDimensaoMedia(linhas, colunas, models, avg);
	svd = calcSVD(models);
	cout<<svd.singularValues()<<endl;

}

int main(int argc, char *argv[]){

	PCA();

	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(windowW, windowH);

	cin.get();
}
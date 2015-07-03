/*COISAS OPENGL*/
#include <Windows.h>//obrigatório incluir para usar o opengl
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
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
#include <math.h>


#include <boost/filesystem.hpp>
#include <boost/foreach.hpp> 
#include <boost/lambda/bind.hpp>
#include <Eigen/Dense> /*Eigen and boost is mostly header-only library. All that you need is to add Eigen path to (MSVC2010):
					 Project Properties -> C/C++ -> General -> Additional Include Directories
					 Let's say you have header Core in folder C:/folder1/folder2/Eigen/, i.e.
					 C:/folder1/folder2/Eigen/Core
					 So you should add path C:/folder1/folder2 to Additional Include Directories.*/
#include <Eigen/SVD>
#include <Eigen/Core>


#define PI 			3.1415926

using namespace std;
namespace fs = boost::filesystem;
//using namespace boost::filesystem;
using namespace boost::lambda;
using namespace Eigen;

/*altura e largura da janela 
de visualização do modelo*/
struct glutWindow{
    int largura;
	int altura;
	string titulo;
 
	float field_of_view_angle;
	float z_near;
	float z_far;
};
glutWindow win;
/*************************/

/*COISAS PCA*/
int linhas, colunas;
MatrixXf listV(1,1);
MatrixXf models(1,1);
MatrixXf avg(1,1);
MatrixXf matrixV(1,1);
MatrixXf matrixVTemp(1,1);
MatrixXf matrixU(1,1);
MatrixXf matrixS(1,1);
MatrixXf matrixCoeffs(1,1);
JacobiSVD<MatrixXf> svd;
/************/
long tamanhoTotalFaces = 0;
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
vector<vertices> novoV;
vector<faces> f;

float xMin, xMax, yMin, yMax, zMin, zMax;
float camX, camY, camZ;
int angX=0, angY=0, angZ=0, roda;
int componente=10;
int deform = 20;

bool leftButton = false;
bool middleButton = false;
int downX = 0;
int downY = 0;
float sphi = 90.0;
float stheta = 45;
float sdepth = 0;

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
				fac.pX-=1;
				fac.pY-=1;
				fac.pZ-=1;
				f.push_back(fac);
				tamanhoTotalFaces++;
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

	for(int i = 0; i< models.rows(); i++){
		for(int j = 0; j< models.cols(); j++){
			sumRow+=models(i,j);
			media = sumRow/models.cols();
					
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

void calcMinMaxValues(vector<vertices> pontos){

	xMin = pontos[0].pX;
	xMax = pontos[0].pX;
	yMin = pontos[0].pY;
	yMax = pontos[0].pY;
	zMin = pontos[0].pZ;
	zMax = pontos[0].pZ;

	for (size_t i=0; i<pontos.size(); i++){
		if(pontos[i].pX<xMin)
			xMin = pontos[i].pX;
		if(pontos[i].pY<yMin)
			yMin = pontos[i].pY;
		if(pontos[i].pZ<zMin)
			zMin = pontos[i].pZ;
		if(pontos[i].pX>xMax)
			xMax = pontos[i].pX;
		if(pontos[i].pY>yMax)
			yMax = pontos[i].pY;
		if(pontos[i].pZ>zMin)
			zMax = pontos[i].pZ;
	}

	camX = (xMin+xMax)/2;
	camY = (yMin+yMax)/2;
	camZ = (zMin+zMax)/2;
}

void varyComponent(){
	float contrib = (deform*sqrt(matrixS(componente,0)))/40.0;
	for(int i=0; i<matrixCoeffs.rows();i++){
		matrixVTemp(i,0) = avg(i,0) + (matrixCoeffs(i,componente)*contrib);
		//cout<<matrixVTemp(i,0)<<endl;
	}
	
	for(int i=0; i<matrixVTemp.rows()/3;i++){
		vert.pX = matrixVTemp(i,0);
		vert.pY = matrixVTemp(i+1411,0);
		vert.pZ = matrixVTemp(i+1411*2,0);	
		novoV.push_back(vert);
	}


	calcMinMaxValues(novoV);

}

void PCA(){

	string nomeFolder = "D:\\MestradoUFES\\projetoMestrado\\FacesMorfologicas\\male10K_1";
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
	preencheListV(linhas, colunas, listV, nomeFolder); //preencho a lista de vertices
	models = listV.transpose();
	calcMedia(linhas,colunas, models, avg); //calcula modelo médio
	subtraiDimensaoMedia(linhas, colunas, models, avg);
	svd = calcSVD(models.transpose());

	cout<<"U "<<svd.matrixU().rows()<<" X "<<svd.matrixU().cols()<<endl;
	cout<<"S "<<svd.singularValues().rows()<<" X "<<svd.singularValues().cols()<<endl;
	cout<<"V "<<svd.matrixV().rows()<<" X "<<svd.matrixV().cols()<<endl;
	

	matrixU.resize(svd.matrixU().rows(),svd.matrixU().cols());
	matrixS.resize(svd.singularValues().rows(),svd.singularValues().rows());
	matrixCoeffs.resize(svd.matrixV().rows(),svd.matrixV().cols());
	matrixVTemp.resize(svd.matrixV().rows(),1);
	matrixU = svd.matrixU();
	matrixS = svd.singularValues();
	matrixCoeffs = svd.matrixV();
	cout<<avg(0,0)<<endl;
	////cout<<matrixS.rows()<<" "<<matrixS.cols()<<endl;
	//calcMinMaxValues(v);
	varyComponent();
	
}


void drawOBJ(){

	GLfloat vE[3] = {0,0,0}, w[3] = {0,0,0}, n[3] = {0,0,0}, lenN;

	for(int i=0;i<f.size();i++) {

		w[0] =  novoV[f[i].pX].pX - novoV[f[i].pY].pX,
		w[1] =  novoV[f[i].pX].pY - novoV[f[i].pY].pY,
		w[2] =  novoV[f[i].pX].pZ - novoV[f[i].pY].pZ,

		vE[0] =  novoV[f[i].pZ].pX - novoV[f[i].pY].pX,
		vE[1] =  novoV[f[i].pZ].pY - novoV[f[i].pY].pY,
		vE[2] =  novoV[f[i].pZ].pZ - novoV[f[i].pY].pZ,

		n[0] =  (vE[1] * w[2]) - (vE[2] * w[1]),
		n[1] =  (vE[2] * w[0]) - (vE[0] * w[2]),
		n[2] =  (vE[0] * w[1]) - (vE[1] * w[0]),

		lenN = sqrt((n[0]*n[0]) + (n[1]*n[1]) + (n[2]*n[2]));

		n[0] = n[0]/lenN;
		n[1] = n[1]/lenN;
		n[2] = n[2]/lenN;

		glBegin(GL_TRIANGLES);
			glNormal3f( n[0], n[1], n[2] );
			glVertex3f(novoV[f[i].pX].pX, novoV[f[i].pX].pY, novoV[f[i].pX].pZ);
			glVertex3f(novoV[f[i].pY].pX, novoV[f[i].pY].pY, novoV[f[i].pY].pZ);
			glVertex3f(novoV[f[i].pZ].pX, novoV[f[i].pZ].pY, novoV[f[i].pZ].pZ);
		glEnd();
	}
}

void Desenha() {

	float angRad = stheta*(PI/180);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	//varyComponent();
	gluLookAt( 2*camX,2*camY,2*camZ,
			  camX,camY, camZ,
			  0,1,0);//ok para o modelo puro

	glPushMatrix();
		
		glTranslatef(camX,camY,camZ); // 3. Translate to the object's position.
		
		glTranslatef(0.0,0.0,-sdepth);
		glRotatef(-stheta,1.0,0.0,0.0); // 2. Rotate the object.
		glRotatef(sphi,0.0,0.0,1.0); // 2. Rotate the obje
		glTranslatef(-camX,-camY,-camZ); // 1. Translate to the origin.
		
		drawOBJ();

	glPopMatrix();
	glutSwapBuffers();
}

void reshape (int w, int h) {

    glViewport (0, 0, (GLsizei)w, (GLsizei)h);

}

void Inicializa() {
    
	glMatrixMode(GL_PROJECTION);
	glViewport(0, 0, win.largura, win.altura);
	GLfloat aspect = (GLfloat) win.largura / win.altura;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	gluPerspective(win.field_of_view_angle, aspect, win.z_near, win.z_far);
    glMatrixMode(GL_MODELVIEW);
    glShadeModel( GL_SMOOTH );
    glClearColor( 0.0f, 0.1f, 0.0f, 0.5f );
    glClearDepth( 1.0f );
    glEnable( GL_DEPTH_TEST );
    glDepthFunc( GL_LEQUAL );
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
 
    GLfloat amb_light[] = { 0.1, 0.1, 0.1, 1.0 };
    GLfloat diffuse[] = { 0.6, 0.6, 0.6, 1 };
    GLfloat specular[] = { 0.7, 0.7, 0.3, 1 };
    glLightModelfv( GL_LIGHT_MODEL_AMBIENT, amb_light );
    glLightfv( GL_LIGHT0, GL_DIFFUSE, diffuse );
    glLightfv( GL_LIGHT0, GL_SPECULAR, specular );
    glEnable( GL_LIGHT0 );
    glEnable( GL_COLOR_MATERIAL );
    glShadeModel( GL_SMOOTH );
    glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE );
    glDepthFunc( GL_LEQUAL );
    glEnable( GL_DEPTH_TEST );
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0); 
}

//Callback de evento de movimentação do mouse
void motion(int x, int y){
    
	if (leftButton){
		sphi+=(float)(x-downX)/4.0;
		stheta+=(float)(downY-y)/4.0;
	} // rotate
	if (middleButton){
		sdepth += (float)(downY - y) / 10.0;
		cout<<sdepth<<endl;
	} // scale
	glutPostRedisplay();
    downX = x;
    downY = y;
}

//Callback de eventos do mouse
void mouse(int button, int state, int x, int y){ 
   
	downX = x;
	downY = y;
	leftButton = ((button==GLUT_LEFT_BUTTON)&&(state==GLUT_DOWN));
	middleButton = ((button == GLUT_MIDDLE_BUTTON) &&  (state == GLUT_DOWN));
	
	glutPostRedisplay();
}

//Callback de eventos de teclado
void Teclado(unsigned char key, int x, int y) {

    switch (key) {
        case 'q':
            roda=0;
			angX++;
            break;
		case 'w':
            roda=1;
			angY++;
            break;
		case 'e':
            roda=2;
			angZ++;
            break;
		case 'a':
			if(componente>=0)
				componente++;
			cout<<"Comp "<<componente<<endl;
            break;
		case 's':
			if(componente<=20)
				componente--;
			cout<<"Comp "<<componente<<endl;
            break;
		case 'z':
			if(deform>=-20)
				deform++;
			cout<<"Deform "<<deform<<endl;
            break;
		case 'x':
			if(deform<=20)
				deform--;
			cout<<"Deform "<<deform<<endl;
            break;
	}
}

int main(int argc, char *argv[]){

	win.largura = 640;
	win.altura = 480;
	win.titulo = "Model Viewer";
	win.field_of_view_angle = 45;
	win.z_near = 0.00001f;
	win.z_far = 500.0f;	

	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(win.largura,win.altura);	
	glutCreateWindow(win.titulo.c_str());
	glutReshapeFunc (reshape);
	glutDisplayFunc(Desenha);
	glutIdleFunc(Desenha);
	glutKeyboardFunc(Teclado);
	glutMouseFunc(mouse);
    glutMotionFunc(motion);

	PCA();

	Inicializa();
	glutMainLoop();
	cin.get();
}


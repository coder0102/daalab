#include<stdio.h>
#include<GL/glut.h>
int x1,y1,x2,y2;
void draw_pixel(int x, int y)
{   glColor3f(0.0,0.0,1.0);
    glPointSize(5);
    glBegin(GL_POINTS);
    glVertex2i(x,y);
    glEnd();  
 }
void Bresenham()
{ glClear(GL_COLOR_BUFFER_BIT);
  glClearColor(1.0,1.0,1.0,1.0);
  draw_line(x1,y1,x2,y2);
   glColor3f(1.0,0.0,0.0);
    glBegin(GL_LINES);
    glVertex2i(x1,y1);
     glVertex2i(x2,y2);
    glEnd();
  glFlush();
}
void myinit()
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0,100,0,100);
  glMatrixMode(GL_MODELVIEW);
}
 
 void draw_line(int x1,int y1, int x2, int y2)
{
    int dx, dy, i, e;
    int incx, incy, inc1, inc2;
    int x,y;
    dx = x2-x1;
    dy = y2-y1;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    incx = 1;
    if (x2 < x1) incx = -1;
    incy = 1;
    if (y2 < y1) incy = -1;
    x = x1; y = y1;

    if (dx > dy) //slope lessthan 1
    {
        	draw_pixel(x, y);
        	e = 2 * dy-dx;// initial decision parameter
       		inc1 = 2*(dy-dx);//upper pixel
        	inc2 = 2*dy;  // lower pixel
        	for (i=0; i<dx; i++)
		 {
            	 	if (e >= 0)
				{y += incy;
				e += inc1;} //selection of upper pixel
	        	else
                 		e += inc2; //selection of Lower pixel

           		x += incx;
               		draw_pixel(x, y);
       	     	}
    } 
   else 
   {
        draw_pixel(x, y);
        e = 2*dx-dy;
        inc1 = 2*(dx-dy);
        inc2 = 2*dx;
    	    for (i=0; i<dy; i++) 
	   {
         	if (e >= 0)
         		 {
 		        x+= incx; //upper pixel
	    		e += inc1;
                         }
         	else
          	 e += inc2;
       		 
            y += incy;
            draw_pixel(x, y);
     	 }
   }
}
void main(int argc,char ** argv)
{ printf("Enter the endpoints of the line segment");
  scanf("%d%d%d%d",&x1,&y1,&x2,&y2);
  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB);
  glutInitWindowSize(500,500);
  glutInitWindowPosition(100,100);
  glutCreateWindow("Bresenham Line Algorithm");
  glutDisplayFunc(Bresenham);
  myinit();
  glutMainLoop();
}


LAB 2
#include<stdio.h>
#include<GL/glut.h>
int n;
typedef GLfloat point2[2];
point2 v[3]= {{-2,-1}, {2,-1},{0,1}};
void triangle( GLfloat *a, GLfloat *b, GLfloat *c)
/* display one triangle  */
{
      glVertex2fv(a); 
      glVertex2fv(b);  
      glVertex2fv(c);

}
void divide_triangle(GLfloat *a, GLfloat *b, GLfloat *c, int m)
{
/* triangle subdivision using vertex numbers */
    point2 v0, v1, v2;
    int j;
if(m>0)
     {
        for(j=0; j<2; j++) v0[j]=(a[j]+b[j])/2;
        for(j=0; j<2; j++) v1[j]=(a[j]+c[j])/2;
        for(j=0; j<2; j++) v2[j]=(b[j]+c[j])/2;
   divide_triangle(a, v0, v1, m-1);
   divide_triangle(c, v1, v2, m-1);
   divide_triangle(b, v2, v0, m-1);
    }
  else(triangle(a,b,c));
 /* draw triangle at end of recursion */
}
void display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_TRIANGLES);
    divide_triangle(v[0], v[1], v[2], n);
    glEnd();
    glFlush();
}
void myinit()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-2.0, 2.0, -2.0, 2.0);
    glMatrixMode(GL_MODELVIEW);
    glClearColor (1.0, 1.0, 1.0,1.0);
    glColor3f(1.0,0.0,0.0);
}
int main(int argc, char **argv)
{
   printf("Enter the number of divisions\n");
	 scanf("%d",&n);
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB);
   glutInitWindowSize(500, 500);
   glutCreateWindow("2D Gasket");
   glutDisplayFunc(display);
	 myinit();
   glutMainLoop();
return 0;
   }


LAB 3
#include <stdlib.h>
#include <stdio.h>
#include <GL/glut.h>
typedef GLfloat point[3]; //Three coordinates x, y,z values
point v[]={{-1.0,-0.5,0.0},{1.0,-0.5,0.0},{0.0,1.0,0.0}, {0.0,0.0,1.0}}; // 4 vertices
GLfloat colors[4][3]={{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0},{1.0,1.0,0.0}};// color for four faces of tetrahedron
int n;// no of divisions
void triangle(point a,point b,point c)   // creating a face of tetrahedron
{
glBegin(GL_POLYGON);
glVertex3fv(a);                                                 
glVertex3fv(b);
glVertex3fv(c);
glEnd();
}
void tetra(point a,point b,point c,point d) // representation of the tetrahedron
{
glColor3fv(colors[0]);
triangle(a,b,c);    //  face-1
glColor3fv(colors[1]);
triangle(a,c,d);      //face-2
glColor3fv(colors[2]);
triangle(a,d,b);      // face-3
glColor3fv(colors[3]);
triangle(b,d,c);     // face-4
}
void divide_tetra(point a,point b,point c,point d,int m) // perform division
{
point mid[6];
int j;
if(m>0)
{
for(j=0;j<3;j++)  // generating the bisetors for each edge
{                                                             
mid[0][j]=(a[j]+b[j])/2.0;  // mid0 between a and b
mid[1][j]=(a[j]+c[j])/2.0;  // mid1 between a and c                   
mid[2][j]=(a[j]+d[j])/2.0;  // mid2 between a and d
mid[3][j]=(b[j]+c[j])/2.0;  //// mid3 between b and c
mid[4][j]=(c[j]+d[j])/2.0;  // mid4 between c and d
mid[5][j]=(b[j]+d[j])/2.0;  // mid5 between b  and d
}
divide_tetra(a,mid[0],mid[1],mid[2],m-1); // apex tetrahedron
divide_tetra(mid[0],b,mid[3],mid[5],m-1); // left corner
divide_tetra(mid[1],mid[3],c,mid[4],m-1); // front 
divide_tetra(mid[2],mid[5],mid[4],d,m-1); // right
}
else
tetra(a,b,c,d);// creating the tetrahedron
}
void display()
{
glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
glClearColor(1.0,1.0,1.0,1.0);
divide_tetra(v[0],v[1],v[2],v[3],n);
glFlush();
}
void myReshape(int w,int h)                                   
{
glViewport(0,0,w,h);
glMatrixMode(GL_PROJECTION);                                    
glLoadIdentity();
if(w<=h)
//glOrtho(-1,1,-1,1,-1,1);
glOrtho(-1.0,1.0,-1.0*((GLfloat)h/(GLfloat)w), 1.0*((GLfloat)h/(GLfloat)w),-1.0,1.0);
else
//glOrtho(-1,1,-1,1,-1,1);
glOrtho(-1.0*((GLfloat)w/(GLfloat)h),1.0*((GLfloat)w/(GLfloat)h),-1.0,1.0,-1.0,1.0);
glMatrixMode(GL_MODELVIEW);
glutPostRedisplay();
}
void main(int argc,char ** argv)
{
printf( "No of Division?: ");
scanf("%d",&n);
glutInit(&argc,argv);
glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB|GLUT_DEPTH);
glutInitWindowSize(500,500);
glutCreateWindow( "3D gasket" );
glutDisplayFunc(display);
glutReshapeFunc(myReshape);
glEnable(GL_DEPTH_TEST);
glutMainLoop();
}


LAB 4
#define BLACK
#include<stdio.h>
#include<math.h>
#include<GL/glut.h>
GLfloat Triangle[3][3]={{100.0,250.0,175.0},{100.0,100.0,300.0},{1.0,1.0,1.0}};
GLfloat rotatement[3][3]={{0},{0},{0}};
GLfloat Result[3][3]={{0},{0},{0}};
GLfloat m=0;
GLfloat n=0;
float theta;

void Triangle()
{
 glColor3f(1.0,0.0,0.0);
 glBegin(GL_LINE_LOOP);
 glVertex2f(Triangle[0][0],Triangle[1][0]);
 glVertex2f(Triangle[0][1],Triangle[1][1]);
 glVertex2f(Triangle[0][2],Triangle[1][2]);
   glEnd();
   }
void display()
{
 glClear(GL_COLOR_BUFFER_BIT);
 Triangle();
 glPushMatrix();
 glTranslatef(m,n,0);
 glRotatef(theta,0,0,1);
 glTranslatef(-m,-n,0);
 Triangle();
 glPopMatrix();
 glFlush();
}
void myinit()
{
 glClearColor(1,1,1,1);
 glColor3f(1,0,0);
 glPointSize(1);
 glMatrixMode(GL_PROJECTION);
 glLoadIdentity();
 gluOrtho2D(0,449,0,499);
 }
 int main(int argc,char **argv)
 {
  int ch;
  printf("enter choice\n 1:Rotation about origin \n 2:rotation about a fixed point\n");
  scanf("%d",&ch);
  switch(ch)
  {
   case 1:printf("enter the rotation angle in degree:");
          scanf("%f",&theta);
          break;
   case 2:printf("enter the fixed point:");
          scanf("%f%f",&m,&n);
          printf("enter the rotation angle:");
          scanf("%f",&theta);       
          break;
  }
  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB);
  glutInitWindowSize(500,500);
  glutInitWindowPosition(0,0);
  glutCreateWindow("Triangle rotaion");	
  glutDisplayFunc(display);
  myinit();
  glutMainLoop();
  return 0;
  }

LAB 5
5.Develop a program to demonstrate 3D transformation on basic objects
#include <stdlib.h>
#include <GL/glut.h>
 GLfloat vertices[][3] = {{-1.0,-1.0,-1.0},{1.0,-1.0,-1.0},{1.0,1.0,-1.0}, {-1.0,1.0,1.0}, {-1.0,-1.0,1.0}  {1.0,-1.0,1.0}, {1.0,1.0,1.0}, {-1.0,1.0,1.0}};
GLfloat normals[][3] = {{-1.0,-1.0,-1.0},{1.0,-1.0,-1.0}, {1.0,1.0,-1.0}, {-1.0,1.0,-1.0},
 {-1.0,-1.0,1.0} {1.0,-1.0,1.0}, {1.0,1.0,1.0}, {-1.0,1.0,1.0}};
GLfloat colors[][3] = {{0.0,0.0,0.0},{1.0,0.0,0.0}, 	{1.0,1.0,0.0}, {0.0,1.0,0.0}, 	{0.0,0.0,1.0},{1.0,0.0,1.0}, {1.0,1.0,1.0}, 	{0.0,1.0,1.0}};

void colorcube(void){
polygon(0,3,2,1);
polygon(2,3,7,6);
polygon(0,4,7,3);
polygon(1,2,6,5);
polygon(4,5,6,7);
polygon(0,1,5,4);
}
static GLfloat theta[]={0.0,0.0,0.0};
static GLint axis =2;
void polygon(int a, int b, int c , int d)
{/* draw a polygon via list of vertices */
 glBegin(GL_POLYGON);
         glColor3fv(colors[a]);
         glNormal3fv(normals[a]);
                glVertex3fv(vertices[a]);
                glColor3fv(colors[b]);
                glNormal3fv(normals[b]);
                glVertex3fv(vertices[b]);
                glColor3fv(colors[c]);
                glNormal3fv(normals[c]);
                glVertex3fv(vertices[c]);
                glColor3fv(colors[d]);
                glNormal3fv(normals[d]);
                glVertex3fv(vertices[d]);      
                glEnd();      
}
void display(void)
{/* display callback, clear frame buffer and z buffer,
   rotate cube and draw, swap buffers */
 glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        glRotatef(theta[0], 1.0, 0.0, 0.0);
        glRotatef(theta[1], 0.0, 1.0, 0.0);
        glRotatef(theta[2], 0.0, 0.0, 1.0);
        colorcube();
        glFlush();
        glutSwapBuffers();
void spinCube()
{/* Idle callback, spin cube 2 degrees about selected axis*/ 
	theta[axis] += 0.1;
        if( theta[axis] > 360.0 ) theta[axis] -= 360.0;
        /* display(); */
        glutPostRedisplay();
}
void mouse(int btn, int state, int x, int y)
{/* mouse callback, selects an axis about which to rotate */  
  if(btn==GLUT_LEFT_BUTTON && state ==    	GLUT_DOWN) axis = 0;      if(btn==GLUT_MIDDLE_BUTTON && state == 	GLUT_DOWN) axis = 1;
 if(btn==GLUT_RIGHT_BUTTON && state == 	GLUT_DOWN) axis = 2;
}
void myReshape(int w, int h)
{    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
   if (w <= h)
        glOrtho(-2.0, 2.0, -2.0 * (GLfloat) h / (GLfloat) w,
            2.0 * (GLfloat) h / (GLfloat) w, -10.0, 10.0);
    else
        glOrtho(-2.0 * (GLfloat) w / (GLfloat) h,
            2.0 * (GLfloat) w / (GLfloat) h, -2.0, 2.0, -10.0, 10.0);
    glMatrixMode(GL_MODELVIEW);
}
void main(int argc, char **argv)
{
   glutInit(&argc, argv);
/* need both double buffering and z buffer */
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(500, 500);
    glutCreateWindow("Rotating a Color Cube");
    glutReshapeFunc(myReshape);
    glutDisplayFunc(display);
    glutIdleFunc(spinCube);
    glutMouseFunc(mouse);
    glEnable(GL_DEPTH_TEST);  /* Enable hidden surface--removal */
    glutMainLoop();
}

LAB 6
#include <GL/glut.h>
#include <math.h>
#include <stdlib.h>
const double TWO_PI = 6.2831853;
GLsizei winWidth = 500, winHeight = 500; // Initial display window size.
GLuint regHex; // Define name for display list.
static GLfloat rotTheta = 0.0;

struct scrPt 
{
GLint x, y;
};
static void init (void)
{
struct scrPt hexVertex;
GLdouble hexTheta;
GLint k;
glClearColor (1.0, 1.0, 1.0, 0.0);
/* Set up a display list for a red regular hexagon.
* Vertices for the hexagon are six equally spaced
* points around the circumference of a circle.
*/
regHex = glGenLists (1);
glNewList (regHex, GL_COMPILE);
glColor3f (1.0, 0.0, 0.0);
glBegin (GL_POLYGON);
for (k = 0; k < 6; k++) {
hexTheta = TWO_PI * k / 6;
hexVertex.x = 150 + 100 * cos (hexTheta);
hexVertex.y = 150 + 100 * sin (hexTheta);
glVertex2i (hexVertex.x, hexVertex.y);
}
glEnd ( );
glEndList ( );
}
void displayHex (void)
{
glClear (GL_COLOR_BUFFER_BIT);
glPushMatrix ( );
glRotatef (rotTheta, 0.0, 0.0, 1.0);
glCallList (regHex);
glPopMatrix ( );
glutSwapBuffers ( );
glFlush ( );
}
void rotateHex (void)
{
rotTheta += 3.0;
if (rotTheta > 360.0)
rotTheta -= 360.0;
glutPostRedisplay ( );
}
void winReshapeFcn (GLint newWidth, GLint newHeight)
{
glViewport (0, 0, (GLsizei) newWidth, (GLsizei) newHeight);
glMatrixMode (GL_PROJECTION);
glLoadIdentity ( );
gluOrtho2D (-320.0, 320.0, -320.0, 320.0);
glMatrixMode (GL_MODELVIEW);
glLoadIdentity ( );
glClear (GL_COLOR_BUFFER_BIT);
}
void mouseFcn (GLint button, GLint action, GLint x, GLint y)
{
switch (button) {
case GLUT_MIDDLE_BUTTON: // Start the rotation.
if (action == GLUT_DOWN)
glutIdleFunc (rotateHex);
break;
case GLUT_RIGHT_BUTTON: // Stop the rotation.
if (action == GLUT_DOWN)
glutIdleFunc (NULL);
break;
default:
break;
}
}
int main (int argc, char** argv)
{
glutInit (&argc, argv);
glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB);
glutInitWindowPosition (150, 150);
glutInitWindowSize (winWidth, winHeight);
glutCreateWindow ("Animation Example");
init ( );
glutDisplayFunc (displayHex);
glutReshapeFunc (winReshapeFcn);
glutMouseFunc (mouseFcn);
glutMainLoop ( );
return(0);
}


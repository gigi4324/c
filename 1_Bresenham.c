#include<stdio.h>
#include<GL/glut.h>

int x1,y1,x2,y2;

void draw_pixel(int x, int y){ 
    glColor3f(0.0,0.0,1.0);
    glPointSize(5);
    glBegin(GL_POINTS);
    glVertex2i(x,y);
    glEnd();
}

void Bresenham(){
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(1.0,1.0,1.0,1.0);
    draw_line(x1,y1,x2,y2);
    glColor3f(1.0,0.0,0.0);
    glBegin(GL_LINES);
    glVertex2i(x1,y1);
    glVertex2i(x2,y2);
    glEnd();
    glFlush();
}
    
void myinit(){
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,100,0,100);
    glMatrixMode(GL_MODELVIEW);
}
    
void draw_line(int x1, int y1, int x2, int y2){
    int dx, dy, p, x, y;
    dx=x2-x1;
    dy=y2-y1;
    x=x1;
    y=y1;
    p=2*dy-dx;

    glBegin(GL_POINTS);
    glVertex2i(x,y);

    while(x<x2){
    x++;
    if(p>=0){
        y=y+1;
        p=p+2*dy-2*dx;
    }
    else p=p+2*dy;
    glVertex2i(x,y);}
    glEnd();
    glFlush();
}
    
void main(int argc,char ** argv){ 
    printf("Enter the endpoints of the line segment");
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
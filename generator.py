from random import randint
from matplotlib import pyplot as plt
import matplotlib
import argparse
import numpy as np
import scipy.ndimage
import json
import copy
import operator


VECTORS={0:(0,-1), 1:(1,-1),2:(1,0),3:(1,1),4:(0,1),5:(-1,1),6:(-1,0),7:(-1,-1)}
INV_VECTORS={(0,-1):0, (1,-1):1,(1,0):2,(1,1):3,(0,1):4,(-1,1):5,(-1,0):6,(-1,-1):7}
NUM_NEIGHBORS=8
RIGHT_TURN=2


def sum_vectors(v1,v2):
        return tuple(map(operator.add, v1,v2))

def neighbor((x,y),direction,turns,distance):
    return sum_vectors((x,y),tuple(map(lambda x: x*distance,VECTORS[(direction+turns)%NUM_NEIGHBORS])))

class Image():

    def __init__(self,width,height,start_x,start_y,outside,inside,border,thickness,directions,space,min_width,max_width,min_length,max_length,obj_color):
        self.WIDTH=width
        self.HEIGHT=height
        self.LATTICE=[[outside]*height for i in xrange(width)]
        self.outside=outside        #exterior color
        self.inside=inside          #interior color
        self.border=border          #border color
        self.current_site=(start_x,start_y)  
        self.current_vector=0
        self.old_vector=0
        self.turns=0                #number of turns in the path
        self.end=self.current_site  #location where the path ends
        self.thickness=thickness    #thickness of the path
        self.directions=directions  #allowed directions for turns       
        self.space=space            #required space before path crosses itself
        self.min_width=min_width    #min thickness of path
        self.max_width=max_width    #max whickness of path
        self.min_length=min_length  #min length of a turn
        self.max_length=max_length  #max length of a turn
        self.obj_color=obj_color    #color for the object
        self.object_space=7
        self.prov=(100,90,80)
        self.cumulative=0
        self.path=[]


    #replace old color with new color
    def change(self,old_color,new_color):
        self.LATTICE=map(lambda row: map(lambda s: new_color if s==old_color else s, row), self.LATTICE)

     #True iff (x,y) has valid coordinates
    def is_valid(self,(x,y)):
        return x>=0 and x<self.WIDTH and y>=0 and y<self.HEIGHT                  

    #True iff moving from (x,y) in direction n makes the path intersect itself.
    def crosses(self, (x,y), n):
        return self.is_inside(neighbor((x,y),n,1,1)) and self.is_inside(neighbor((x,y),n,-1,1))

    #True iff (x,y) has valid coordinates but is not in the interior
    def is_not_inside(self,(x,y)):
        return self.is_valid((x,y)) and self.LATTICE[x][y]!=self.inside
                                                       
    #True iff (x,y) is in the interior       
    def is_inside(self,(x,y)):
        return self.is_valid((x,y)) and self.LATTICE[x][y]==self.inside

    #True iff it is possible to move m times from (x,y) in direction (a,b) without the path crossing itself.
    def has_space(self,(x,y), (a,b)):
        n=INV_VECTORS[(a,b)]
        for i in xrange(1,self.space):
            if not self.is_valid((x+i*a,y+i*b)) or self.is_inside((x+i*a,y+i*b)) or self.crosses((x+(i-1)*a,y+(i-1)*b),n):
                return False
        return True

    #True if the neighbor in question is a valid next position
    def valid_neighbor(self,neighbor,direction):
        (x,y)=self.current_site
        return self.is_not_inside(neighbor) and not self.crosses((x,y),direction) and self.has_space(neighbor,VECTORS[direction])

    #Returns a list with all possible locations for next move.
    def valid_neighbors(self):
        neighbors=list(map(lambda i:(neighbor(self.current_site,self.current_vector,i,1),(self.current_vector+i)%8),self.directions))
        return filter(lambda (neighbor,i): self.valid_neighbor(neighbor,i),neighbors)

    #changes the color of a point if the restriction holds
    def set_color(self,(x,y),color,restriction):
        if restriction((x,y)):
            self.LATTICE[x][y]=color

    #Set border of random thickness for the current position
    def fill(self,minimum,maximum,step):
        if self.cumulative>1:
            width=randint(max(self.thickness-1,minimum),min(self.thickness+1,maximum))
            if width!=self.thickness:
                self.cumulative=0
        else:
            width=self.thickness
            self.cumulative+=1
        if step==0:
            width=self.thickness
        for i in range(width):
            for j in {RIGHT_TURN,-RIGHT_TURN}:
                (x,y)=neighbor(self.current_site,self.current_vector,j,i)
                self.set_color((x,y),self.inside,self.is_valid)
                if self.current_vector%RIGHT_TURN==1:
                    self.set_color((x+np.sign(j),y),self.prov,self.is_not_inside)
        self.thickness=width
            
    #fixed the corner when the path changes direction
    def fix_corner(self):
        difference=self.current_vector-self.old_vector
        if abs(difference) in {0,4}:
            return
        reference=self.old_vector
        direction=self.old_vector
        if self.current_vector%2==0:
                reference=self.current_vector
                direction=self.current_vector+4
        for j in range(1,self.thickness-1):
            for i in range(self.thickness):
                for k in {RIGHT_TURN,-RIGHT_TURN}:
                    (x,y)=neighbor(self.current_site,reference,k,i)
                    self.set_color(neighbor((x,y),direction,0,j),self.prov,self.is_valid)

    #Counts how many neighbors of the site (x,y) at a given distance have a specific color.
    def count_nbr_color(self,x,y,color,distance):
        count=0
        for vector in VECTORS.keys():
            (n1,n2)=neighbor((x,y),vector,0,distance)
            if self.is_valid((n1,n2)) and self.LATTICE[n1][n2]==color:
                count+=1
        return count

    #Removes isolated points with border color
    def remove_spots(self):
        for y in xrange(self.HEIGHT):
            for x in xrange(self.WIDTH):
                if self.LATTICE[x][y]==self.border and self.count_nbr_color(x,y,self.inside,1)==0:
                            self.LATTICE[x][y]=self.outside

    #Redefine border to have thickness 1
    def fix_border(self):
        for y in xrange(self.HEIGHT):
            for x in xrange(self.WIDTH):
                if self.LATTICE[x][y]!=self.outside:
                    if self.count_nbr_color(x,y,self.outside,1)>0:
                        if self.count_nbr_color(x,y,self.inside,1)>0 or self.count_nbr_color(x,y,self.prov,1)>0:
                            self.LATTICE[x][y]=self.border
                        else:
                            self.LATTICE[x][y]=self.outside
                    else:
                        self.LATTICE[x][y]=self.inside

    #Create a closed random path with at most n turns             
    def create_random_path(self,n):
        for j in xrange(n):
            self.turns+=1
            valid_nbrs=self.valid_neighbors()
            if len(valid_nbrs)!=0:
                nbr=valid_nbrs[randint(0,len(valid_nbrs)-1)] #choose random neighbor
                self.current_vector=nbr[1]                   #update direction
                side_length=randint(self.min_length,self.max_length) 
                for i in xrange(side_length):                #move in the same direction for random length
                    self.LATTICE[self.current_site[0]][self.current_site[1]]=self.inside
                    self.path.append(self.current_site)
                    self.fill(self.min_width,self.max_width,i)
                    vector=VECTORS[nbr[1]]
                    (x,y)=sum_vectors(self.current_site,vector) 
                    if i==0 and j!=0:
                        self.fix_corner()
                    if i!=side_length-1:                    #check if future position is valid
                        if self.is_not_inside((x,y)) and not self.crosses(self.current_site,nbr[1]) and self.has_space((x,y),vector):
                            self.current_site=(x,y)
                    else:
                        break
                self.old_vector=self.current_vector
                self.current_vector=nbr[1]
            else:
                self.end=self.current_site
                break

    #display image         
    def display(self):
        image = np.array(self.LATTICE, dtype=np.uint8)
        plt.imshow(image, interpolation='none')
        plt.show()

    #Returns a set with all neighbor sites of (x,y)
    def get_neighbors(self,(x,y)):
        neighbors=set()
        for vector in INV_VECTORS.keys():
            nbr=sum_vectors((x,y),vector)
            if self.is_valid(nbr):
                neighbors.add(nbr)
        return neighbors

    #assigns a new color to all neighbors of a point.
    def paint_neighbors(self,(x,y)):
        for vector in INV_VECTORS.keys():
            self.set_color(sum_vectors((x,y),vector),self.obj_color, self.is_valid)

    #draws an object inside of the closed path
    def obj_inside(self):
        (x,y)=self.path[randint(2,len(self.path)-3)]
        self.LATTICE[x][y]=self.obj_color
        self.paint_neighbors((x,y))
        
    #draws an object outside of the closed path
    def obj_outside(self):
        while True:
            y=randint(0,self.HEIGHT-1)
            for a in xrange(min(self.WIDTH-self.end[0],self.end[0])):
                for x in {a+self.WIDTH/2,-a+self.HEIGHT/2}:
                    if self.LATTICE[x][y]==self.outside and self.count_nbr_color(x,y,self.outside,1)==8:
                        self.LATTICE[x][y]=self.obj_color
                        self.paint_neighbors((x,y))
                        return


#Create and display a new image            
def create_image(width,height, start_x, start_y,background, interior, border, path_width, directions, space, \
                 min_width,max_width, min_length, max_length, object_color, min_turns, max_turns, inside):
    count=0
    while count<min_turns:
        image=Image(width,height,start_x,start_y,background,interior,border,path_width,directions,
                 space,min_width,max_width,min_length,max_length,object_color)
        image.create_random_path(max_turns)
        count=image.turns
    image.fix_border()
    image.remove_spots()
    if inside:
        image.obj_inside()
    else:
        image.obj_outside()
    image.change(interior,background)
    image.display()

def create_data_set(width,height, background, interior, border, path_width, directions, space, \
                 min_width,max_width, min_length, max_length, object_color, min_turns, max_turns,\
                    size, filename ):
    images={}
    m=0
    for j in xrange(size):
        start_x=randint(width/3,2*width/3)
        start_y=randint(height/3,2*height/3)
        count=0
        while count<min_turns:
            image=Image(width,height,start_x,start_y,background,interior,border,path_width,directions,
                 space,min_width,max_width,min_length,max_length,object_color)
            image.create_closed_path(max_turns)
            count=image.turns
        image.fix_border()
        image.remove_spots()
        u=randint(0,1)        #randomly choose if the object is inside or outside
        r=(0,0)
        if u==0:
            image.obj_inside()
            image.change(interior,background)
            r=(image.lattice(),1)
        else:
            image.obj_outside()
            image.change(interior,background)
            r=(image.lattice(),0)
        images[m]=r
        m+=1
    with open(filename,"w") as f:
        json.dump(images,f)
        
                           

#Examples: data_set(30,30,0,2,1,5,{1,2,3,4,5,6,7,8},3,3,5,5,15,-1,3,3,100000,"dataset.json")
#          create_image(200,200,60,60,(0,0,0),(0,250,250),(216,100,123),7,{1,2,3,4},18,5,8,35,40,(1,196,255),20,40,True)


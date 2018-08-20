include("./mehrgitter.jl")
using Mehrgitter
using MPI

function main(n::Int64, k::Int64, iterationen::Int64, toleranz::Float64, param::Float64, maxiter::Float64)
   
    #es wird herausgefynden wie viele prozessore und welche id diese haben
    id = MPI.Comm_rank(MPI.COMM_WORLD)
    num_procs = MPI.Comm_size(MPI.COMM_WORLD)
    
    globx=1
    globy=1
    anzx=0
    anzy=0
    funktion(p,y,k)=sin(k*pi*p)*sin(k*pi*y)
    funktion2(p,y,k)=2*k*k*pi*pi*sin(k*pi*p)*sin(k*pi*y)

    #die prozessorenanzahl darf nicht null sein und muss eine Quadratzahl oder 2er Potenz sein(mit bitweise ueberpruefung)
    if num_procs==0 || num_procs&(num_procs-1)!=0 
        if convert(Int64, floor(sqrt(num_procs)))^2!=num_procs 
            return -1
        end
    end
    
    #Ab hier werden die Prozessoren aufgeteilt, das heisst genau, dass jedem Prozessor eine matrixgroesse und ein globaler Index zugeordnet werden, abhaengig von seiner id
    #Der Prozessor wird folgedermassen aufgeteilt:
    #   --> als erstes wird ein vertikaler Schnitt in der Mitte gemacht, bei ungerader Anzahl wird dem oberen Block eine Zeile mehr zugeordnet
    #   --> dann wird ein horiyontal Schnitt in der Mitte gemacht, bei ungerader Anzahl bekommt der linke Block eine Spalte mehr
    #   --> dies wird fuer jeden Abschnitt solange gemacht, wie Prozessoren da sind
    # Wenn wir also eine 7x7 Matrix haben und vier Prozessoren, ist die Matrix folgendermassen aufgeteilt:(4,4)(4,3)(3,4)(3,3)(nach den id; ersten beiden oben , zweiten beiden unten)
    #Abhaengig von der Prozessorenanzahl wir bei einer quadratzahl die Anzahl der x&yprocs festgestellt, indem man Wurzel zieht
    #Bei Zweierpotenzen ist yprocs=xprocs*2
    #Globale Koordinate: (globx|globy)
    #Lokale Groesse in x richtung anz_x und in y richtung anz_y
    if floor(sqrt(num_procs))^2==num_procs || num_procs==1
        xprocs=sqrt(num_procs)
        yprocs=xprocs
    else
        xprocs=sqrt(convert(Int64, num_procs/2))
        yprocs=xprocs*2
    end
    #Hier wird der globale xIndex festgelegt
    globx=convert(Int64, ceil(n/yprocs)*(id%yprocs)+1)
    globy=convert(Int64, floor(id/yprocs)*ceil(n/xprocs)+1)
    #Die Anzahl der Y Dimension wird festgestelt, indem man n/xprocs teilt, wenn dieser Abschnitt ein Randabschnitt und nicht der einzige Abschnitt ist, werden noch zeilen abgezogen
    anzy=convert(Int64, ceil(n/xprocs))
    if num_procs-yprocs<=id
        anzy=n-globy+1
    end
    #Hier wird der globale yIndex festgelegt
    #Die Anzahl der Y Dimension wird festgestelt, indem man n/yprocs teilt, wenn dieser Abschnitt ein Randabschnitt und nicht der einzige Abschnitt ist, werden noch spalten abgezogen
    anzx=convert(Int64, ceil(n/yprocs))
    if id%yprocs==yprocs-1 && yprocs!=1
        anzx=n-globx+1
    end
    #Levelnachbarn werden gefunden
    #1.Anzahl Level feststellen:
    level=convert(Int64,log2(n+1))
    #Mache f und v
    srand(12345)
    varray=Array{Array{Float64,2}}(level)
    #varray[level]=abs.(sin.(rand(anzy,anzx)))
    farray=Array{Array{Float64,2}}(level)
    #farray[level]=zeros(anzy,anzx)
    #Fuer jeden Prozessor gibt es einen Representantenunten rechts und oben links, welcher unten rechts bei jedem Prozessor liegt(x|y), es wird angenommen das das Feld zwei grade Matrixgroessen hat
    unten_representant_x=globx+anzx-1
    unten_representant_y=globy+anzy-1
    oben_representant_x=-1
    oben_representant_y=-1
    #2. Array fuer Nachbarn anlegen und mit -1 belegt, damit sich Randwerte unterscheiden, Ein 3*3 Array wird angelegt um auch mit vollstaendigen Stencils besser arbeiten zu koennen
    nachbarn = -1*ones(Int64,3,3,level)
    loesung=Array{Float64,2}(1,1)
    #Abstand auf groebstem Level  
    for i in 1:level
        #Sprungweite bestimmen(also wie weit sind 2 Punkte von einander weg)
        sprung=2^(i-1)
        if anzx%2==1
            unten_representant_x=globx+anzx-sprung
        end
        if anzy%2==1
            unten_representant_y=globy+anzy-sprung
        end
        oben_representant_y=convert(Int64,unten_representant_x-sprung*(floor((unten_representant_x/sprung)-(globy/sprung))))
        oben_representant_x=convert(Int64,unten_representant_y-sprung*(floor((unten_representant_y/sprung)-(globx/sprung))))
        #Gehe nur in diese if Bedingung, wenn der Prozessor kein Loch ist
        if unten_representant_x%sprung==0 && unten_representant_y%(sprung)==0 && unten_representant_x>=globx && unten_representant_y>=globy
            nachbarn[2,2,level+1-i]=id
            #rechts?
            if unten_representant_x+(sprung)<=n
                nachbarn[2,3,level+1-i]=convert(Int64,id+ceil((sprung)/anzx))
            end
            #links
            if oben_representant_x-(sprung)>=1
                nachbarn[2,1,level+1-i]=convert(Int64,id-ceil((sprung)/anzx))
            end
            #unten?
            if unten_representant_y+(sprung)<=n
                nachbarn[3,2,level+1-i]=convert(Int64,id+ceil((sprung)/anzy)*yprocs)
            end
            #oben
            if oben_representant_y-(sprung)>=1
                nachbarn[1,2,level+1-i]=convert(Int64,id-ceil((sprung)/anzy)*yprocs)
            end
            #links oben
            if nachbarn[1,2,level+1-i] != -1 && nachbarn[2,1,i] != -1
                nachbarn[1,1,level+1-i] = convert(Int64,id-ceil((sprung)/anzy)*yprocs-ceil((sprung)/anzx))
            end
            #rechts oben
            if nachbarn[1,2,level+1-i] != -1 && nachbarn[2,3,i] != -1
                nachbarn[1,3,level+1-i] = convert(Int64,id-ceil((sprung)/anzy)*yprocs+ceil((sprung)/anzx))
            end
            #links unten
            if nachbarn[3,2,level+1-i] != -1 && nachbarn[2,1,i] != -1
                nachbarn[3,1,level+1-i] = convert(Int64,id+ceil((sprung)/anzy)*yprocs-ceil((sprung)/anzx))
            end
            #rechts unten
            if nachbarn[3,2,level+1-i] != -1 && nachbarn[2,3,i] != -1
                nachbarn[3,3,level+1-i] = convert(Int64,id+ceil((sprung)/anzy)*yprocs+ceil((sprung)/anzx))
            end
            #Der letzte Array ist der mit dem feinsten Gitter --> Matrixstruktur fuer f und v festlegen
            lok_size_per_lvl_x=convert(Int64,floor((unten_representant_x-oben_representant_x+sprung)/sprung))
            lok_size_per_lvl_y=convert(Int64,floor((unten_representant_y-oben_representant_y+sprung)/sprung))
            if i==1
                #HIER KANN STARTWERT, ERGEBNIS UND LOESUNG EINGESTELLT WERDEN
                varray[level+1-i]=fourier(funktion,n,k,oben_representant_y,oben_representant_x,lok_size_per_lvl_y,lok_size_per_lvl_x, sprung)
                farray[level+1-i]=zeros(lok_size_per_lvl_y,lok_size_per_lvl_x)
                #varray[level+1-i]=random_matrix(n,[oben_representant_y,oben_representant_x],[lok_size_per_lvl_y,lok_size_per_lvl_x])
                loesung=zeros(lok_size_per_lvl_y,lok_size_per_lvl_x)
                #farray[level+1-i]=fourier(funktion2,n,2,oben_representant_y,oben_representant_x,lok_size_per_lvl_y,lok_size_per_lvl_x, sprung)
                #loesung=fourier(funktion,n,2,oben_representant_y,oben_representant_x,lok_size_per_lvl_y,lok_size_per_lvl_x, sprung)
            else
                varray[level+1-i]=zeros(lok_size_per_lvl_y,lok_size_per_lvl_x)
                farray[level+1-i]=zeros(lok_size_per_lvl_y,lok_size_per_lvl_x)
            end
        else
            varray[level+1-i]=Array{Float64,2}(1,1)
            farray[level+1-i]=Array{Float64,2}(1,1)
        end
    end
    #A wird erzeugt
    aarray =Array{Int64,2}(level,5)
    array = [1,1,-4,1,1]
    for i in 1:5
        for j in 1:level
            aarray[j,i]=-(2^j)^2*array[i]
        end
    end
    #Es wird produktiv geloest: Av=f
    zeit=0
    #jacobi!(id, num_procs, varray[level],aarray[level,:],farray[level],0.0,param,iterationen,nachbarn[:,:,level])
    MPI.Barrier(MPI.COMM_WORLD)
    #t=time()
    t=time()
    schritte, x, residuum=mehrgittermethode_rekursiv(aarray,varray, farray, iterationen, level, toleranz, [globx, globy], id, num_procs, nachbarn, param, maxiter,n)
    t=time()-t
    #t=time()-t
    #Maximum bestimmen
    #max=maximum(abs.(x))
    max=maximum(abs.(loesung-x))
    y=MPI.Reduce(max, MPI.MAX, 0, MPI.COMM_WORLD)
    return y, t, schritte, residuum
end

MPI.Init()
#Matrixgroesse
nmin=7
nmax=1000
#Fouriermode
kmin=3
kmax=3
#Iterationsanzahl im Jacobi
iterationen=3
#Genauigkeit
fehler=0.0
#Parameter fuer Gewichteten Jacobi(optimal)
param=4/5
#Anzahl der Mehrgitteriterationen
miniter=20.0
maxiter=miniter

id = MPI.Comm_rank(MPI.COMM_WORLD)
num_procs = MPI.Comm_size(MPI.COMM_WORLD)

# Eingabe der Daten ist optional und kein muss
#Der Name der Eingabedatei muss angegeben werden
eingabe=open(ARGS[1],"r")
datei=open(string(ARGS[2],"_",num_procs,".txt"),"w")
lines = readlines(eingabe)
for line in lines
    line=split(line)
    if line[1] != "#"
        if line[1]=="nmin"
            nmin = parse(Int64,line[3])
        elseif line[1]=="nmax"
            nmax = parse(Int64,line[3])
        elseif line[1]=="kmin"
            kmin = parse(Int64,line[3])
        elseif line[1]=="kmax"
            kmax = parse(Int64,line[3])
        elseif line[1]=="iterationen"
            iterationen = parse(Int64,line[3])
        elseif line[1]=="error"
            fehler = parse(Float64,line[3])
        elseif line[1]=="param"
            param = parse(Float64,line[3])
        elseif line[1]=="miniter"
            miniter = parse(Float64,line[3])
        elseif line[1]=="maxiter"
            maxiter = parse(Float64,line[3])
        end
    end
end
close(eingabe)

#Ueberpruefung
if(kmin>kmax)
    help=kmin
    kmin=kmax
    kmax=help
end
if(kmax>nmax)
    kmax=nmax-1
end
if(kmin<=0)
    kmin=1
end
if(iterationen<=0)
    iterationen=1
end
if(fehler<0)
    fehler=0.0
end
if(nmin in [3,7,15,31,63,127,255,511,1023,2047,4095,8191,16383])

else
    nmin=7
end
if miniter>maxiter
    help=miniter
    miniter=maxiter
    maxiter=help
end

while nmin<=nmax
    k=kmin
    while k<=kmax
        iter=miniter
        while iter<=maxiter
            #println(n)
            error, t2, schritte, residuum =main(nmin,k,iterationen, fehler, param, iter)
            t=0
            t=MPI.Reduce(t2, MPI.MAX, 0, MPI.COMM_WORLD)
            if(id==0)
                println(nmin, " ", k," ", schritte," ",residuum)
                #Anzahl Procs, Matrixgroesse n, Fouriermode k, Iterationen, Abstand zur genauen Loesung, Zeit die das gesamte Mehrgitter braucht, Residuum, gegebener Fehler
                write(datei,string(num_procs)," ",string(nmin)," ",string(k)," ",string(schritte)," ",string(error)," ",string(t2)," ",string(residuum)," ", string(fehler), " \n")
            end
            iter=iter+1
        end
        k=k+10
    end
    nmin=2*(nmin+1)-1
end
close(datei)
MPI.Finalize()

module Mehrgitter
__precompile__()

export mehrgittermethode_rekursiv, fourier, random_matrix, jacobi!, redblackgs!

    using MPI

    #Ich arbeite mit einer geraenderten Matrix bei diesem Verfahren,
    #dies macht die Implementierung leichter.
    function fuege_Rand_hinzu(u::Array{Float64,2})
        N1=size(u)[1]
        N2=size(u)[2]
        matrix=zeros(N1+2,N2+2)
        matrix[2:N1+1,2:N2+1] = u
        return matrix
    end

    #Auch hier arbeite ich mit einer geraenderten Matrix
    #Die Matrix wird normal restringiert --> xneu also das kleinere Level wird veraendert
    function restriktion!(x::Array{Float64,2},xneu::Array{Float64,2},koord::Array{Int64,1}, level::Int64, id, num_procs, nachbarn, nachbarn_grob,n)
        x=fuege_Rand_hinzu(x)
        altk=3
        altl=3
        if size(x)[1]==3 #&& floor(koord[1]/2^(log2(n+1)-level))%2==1
            altk=2
        end
        if size(x)[2]==3 #&& floor(koord[1]/2^(log2(n+1)-level))%2==1
            altl=2
        end
        halo_oben_unten!(id, num_procs,x,nachbarn)
        halo_rechts_links!(id, num_procs,x,nachbarn)
        for i in altk-1:size(x)[1]-1
            for j in altl:2:size(x)[2]-1
                x[i,j]=0.5*x[i,j]+0.25*x[i,j-1]+0.25*x[i,j+1]
            end
        end
        halo_rechts_links!(id, num_procs,x,nachbarn)
        halo_oben_unten!(id, num_procs,x,nachbarn)
        k=altk
        if nachbarn_grob[2,2]==id
            for i in 1:size(xneu)[1]
                l=altl
                for j in 1:size(xneu)[2]
                    xneu[i,j]=0.5*x[k,l]+0.25*x[k-1,l]+0.25*x[k+1,l]
                    l=l+2
                end
                k=k+2
            end
        end
        return xneu
    end

    #Die Matrix wird beim einsetzen geraendert und es erfolgt eine Prolongation# die koordinaten muessen die von xneu sein
    function prolongation(x::Array{Float64,2},xref::Array{Float64,2},koord::Array{Int64,1}, level, id, num_procs, nachbarn, nachbarn_grob)
        xneu=fuege_Rand_hinzu(zeros(xref))
        altk=3
        altl=3
        if size(xneu)[1]==3
            altk=2
        end
        if size(xneu)[2]==3
            altl=2
        end
        if nachbarn_grob[2,2]==id
            k=altk
            for i in 1:size(x)[1]
                l=altl
                for j in 1:size(x)[2]
                    xneu[k,l]=x[i,j]
                    l=l+2
                end
                k=k+2
            end
        end
        halo_rechts_links!(id, num_procs,xneu,nachbarn)
        #1. Zeilen in denen schon Werte stehen werden bearbeitet
        if (altk==3&&altl==3) || xneu[altk,altl]==0.0
            for i in collect(altk:2:size(xneu)[1]-1)
                for j in collect(2:2:size(xneu)[2]-1)
                    xneu[i,j]=0.5*xneu[i, j-1]+0.5*xneu[i, j+1]
                end
            end
        end
        halo_oben_unten!(id, num_procs,xneu,nachbarn)
        #2. Zeilen in denen noch keine Werte stehen werden bearbeitet rechts unten ist ein sonderfall sowie wenn auf allen nur 1 wert ist
        alf=0
        if altk==2 &&altl==2 && floor(koord[2]/2^(log2(size(xneu)[1]+1)-level))%2==0 && nachbarn[3,2]!=-1
            alf=1
        end
        for i in collect(2+alf:2:size(xneu)[1]-1)
            for j in collect(2:1:size(xneu)[2]-1)
                xneu[i,j]=0.5*xneu[i-1, j]+0.5*xneu[i+1, j]
            end
        end
        xneu=xneu[2:end-1, 2:end-1]
        return xneu
    end

    #Rekursive Mehrgitterimplementierung
    #Glaetter: Jacobi
    #
    function mehrgitter!(A::Array{Int64,2},varray::Array{Array{Float64,2},1}, farray::Array{Array{Float64,2},1},iterationen::Int64, level::Int64, koord::Array{Int64,1}, id::Int64, num_procs::Int64, nachbarn, param::Float64,n)
        #Rekursiver Anker
        if nachbarn[2,2,level] ==id
            if level==1
                varray[level,1,1]=farray[level,1,1]/A[level,3]
                return 0
            end
            level=level-1
            #Vorglaetten
            #HIER MUSS GEANDERT WERDEN WELCHEN GLAETTER MAN WILL
            varray[level+1]=jacobi!(id, num_procs, varray[level+1],A[level+1,:],farray[level+1],0.0,param,iterationen,nachbarn[:,:,level+1])
            #varray[level+1]=redblackgs!(id, num_procs, varray[level+1],A[level+1,:],farray[level+1],0.0,iterationen,nachbarn[:,:,level+1],level+1,koord,n)
            
            #residuum
            r=residuum_mit_halo(A[level+1,:],varray[level+1],farray[level+1],id, num_procs,nachbarn[:,:,level+1])
            #restriktion
            farray[level]=restriktion!(r,farray[level],koord, level, id, num_procs, nachbarn[:,:,level+1],nachbarn[:,:,level],n)
            #rekursion
            mehrgitter!(A,varray,farray,iterationen,level, koord,id,num_procs,nachbarn, param,n)
            #prolongation
            varray[level+1]+=prolongation(varray[level],varray[level+1],koord, level, id, num_procs, nachbarn[:,:,level+1],nachbarn[:,:,level])
            
            #Nachglaetten
            #HIER MUSS GEANDERT WERDEN WELCHEN GLAETTER MAN WILL
            #varray[level+1]=redblackgs!(id, num_procs, varray[level+1],A[level+1,:],farray[level+1],0.0,iterationen,nachbarn[:,:,level+1],level+1,koord,n)
            varray[level+1]=jacobi!(id, num_procs, varray[level+1],A[level+1,:],farray[level+1],0.0,param,iterationen,nachbarn[:,:,level+1])
        end
    end
    

    #Ruft den Mehrgittercode auf. 
    function mehrgittermethode_rekursiv(A::Array{Int64,2},varray::Array{Array{Float64,2},1}, farray::Array{Array{Float64,2},1},iterationen::Int64, level::Int64,toleranz::Float64, koord::Array{Int64,1}, id::Int64, num_procs::Int64, nachbarn, param::Float64, maxiter::Float64,n)
        i=0
        #Residuum wird gebraucht um zu wissen wann man aufhoeren soll
        res=[0.0]
        max=[maximum(abs.(residuum_mit_halo(A[level,:],varray[level],farray[level],id,num_procs, nachbarn[:,:,level])))]
        MPI.Allreduce!(max,res, MPI.MAX, MPI.COMM_WORLD)
        #Residuum bestimmen --> Ist Abbruchbedingung
        while(i<maxiter) #&& toleranz<=res[1])
            
            #println(res[1], " ", toleranz," ",toleranz<=res[1] )
            #Iterationsaufruf
            
            for j in 1:level-1
                varray[j]=zeros(varray[j])
                farray[j]=zeros(farray[j])
            end
            mehrgitter!(A,varray, farray,iterationen, level, koord,id, num_procs,nachbarn, param,n)
            i=i+1
            max[1]=maximum(abs.(residuum_mit_halo(A[level,:],varray[level],farray[level],id,num_procs, nachbarn[:,:,level])))
            MPI.Allreduce!(max,res, MPI.MAX, MPI.COMM_WORLD)
        end
        return i, varray[end], res[1]
    end

    #Momentan gebraucht, da fuer die Berechnung des Residuums ein Rand gebraucht wird und dieser auch aktualisiert werden muss und dies nur im JAcobi der fAll ist aber nicht ausserhalb
    function residuum_mit_halo(A::Array{Int64,1}, x::Array{Float64,2}, b::Array{Float64,2},id, num_procs,nachbarn::Array{Int64,2})
        x=fuege_Rand_hinzu(x)
        b=fuege_Rand_hinzu(b)
        halo_rechts_links!(id, num_procs,x,nachbarn)
        halo_oben_unten!(id, num_procs,x,nachbarn)
        r=zeros(x)
        residuum!(A,x,b,r)
        r=r[2:end-1, 2:end-1]
        x=x[2:end-1, 2:end-1]
        return r
    end

    #u entspricht x oder xalt??==>residuum wird mit x vom vorlauf davor erstellt deswegen muesste das passen?
    #Hier wird das Residuum r=b-Ax berechnet
    #ISt ausgelagert, da mehrere Funktionen das machen muessen
    function residuum!(A::Array{Int64,1}, x::Array{Float64,2}, b::Array{Float64,2}, residuum::Array{Float64,2})
        for i in 2:size(residuum)[1]-1
            for j in 2:size(residuum)[2]-1
                residuum[i,j]=b[i,j]-(A[1]*x[i-1,j]+A[2]*x[i,j-1]+A[3]*x[i,j]+A[4]*x[i,j+1]+A[5]*x[i+1,j])
            end
        end
        
    end

    # 2-Dimensionaler Jacobi Algorithmus
    # Schau in die Doku wenn du nicht weisst wies funktioniert!
    # Rueckgabe aktualisiertes x
    function jacobi!(id::Int64, num_procs::Int64, x::Array{Float64,2},a::Array{Int64,1},b::Array{Float64,2},fehler::Float64, w::Float64,anzahl::Int64,nachbarn::Array{Int64,2})
            #Die zu bearbeitenden Matritzen werden geraendert.
            x=fuege_Rand_hinzu(x)
            b=fuege_Rand_hinzu(b)
            #Die Raender werden mit den Nachbarn getauscht, damit lokales arbeiten Problemlos moeglich ist
            halo_rechts_links!(id, num_procs,x,nachbarn)
            halo_oben_unten!(id, num_procs,x,nachbarn)
            #Residuum: r=b-A*x
            residuum=zeros(x)
            residuum!(a,x,b,residuum)
            anz=0
            #Soll so oft gemacht werden wie vom Programmierer gewuenscht
            #Nachfolgend Jacobi iterativer Loeser, welcher zum glaetten gebraucht wird. NAch jeder Iteration werden die Raender getauscht.
            while(anz<anzahl)
                for i in 2:size(x)[1]-1
                    for j in 2:size(x)[2]-1
                        x[i,j]= x[i,j]+((w/a[3])*residuum[i,j])#WAS WAR DAS HIER?+a[3] ist unschoen aber muss leider sein weil das im Residuum zuviel abgezogen wird
                    end
                end
                anz=anz+1
                halo_rechts_links!(id, num_procs,x,nachbarn)
                halo_oben_unten!(id, num_procs,x,nachbarn)
                residuum!(a,x,b,residuum)
            end
            x=x[2:end-1, 2:end-1]
            return x
    end

    
    function redblackgs!(id::Int64, num_procs::Int64, x::Array{Float64,2},a::Array{Int64,1},b::Array{Float64,2},fehler::Float64, anzahl::Int64, nachbarn::Array{Int64,2}, level::Int64, index::Array{Int64,1},n)
        x=fuege_Rand_hinzu(x)
        b=fuege_Rand_hinzu(b)
        halo_rechts_links!(id, num_procs,x,nachbarn)
        halo_oben_unten!(id, num_procs,x,nachbarn)
        m=0
        p=1
        if size(x)[2]==3 && 2^(log2(n+1)-level)!=1
            stelle=floor(index[1]/2^(log2(n+1)-level))
            if stelle%2==1
                m=1
                p=0
            end
        end
        if size(x)[1]==3 && 2^(log2(n+1)-level)!=1
            stelle=floor(index[2]/2^(log2(n+1)-level))
            if stelle%2==1
                help = m
                m = p
                p = help
            end
        end
        z=0
        while(z<anzahl)
            #Erste Schleife
            berechnung(x,a,b,m)
            halo_rechts_links!(id, num_procs,x,nachbarn)
            halo_oben_unten!(id, num_procs,x,nachbarn)
            berechnung(x,a,b,p)
            halo_rechts_links!(id, num_procs,x,nachbarn)
            halo_oben_unten!(id, num_procs,x,nachbarn)
            z=z+1
        end
        x=x[2:end-1, 2:end-1]
        return x
    end

    #fuer rbgs berechnung gebraucht wegen startwert
    function berechnung(x::Array{Float64,2},a::Array{Int64,1},b::Array{Float64,2},var::Int64)
        for i in 2:(size(x)[1]-1)
            for j in collect(2+((i+var)%2):2:size(x)[2]-1)
                x[i,j]= ((1/a[3])*(b[i,j]-(a[1]*x[i-1,j]+a[2]*x[i,j-1]+a[4]*x[i,j+1]+a[5]*x[i+1,j])))
            end
        end
    end

    #erzeugt einen 2Dimensionalen Array --> Fouriermode
    #Rueckgabe aktualisierte Fouriermode 
    function fourier(f::Function, N::Int64,k::Int64, globx::Int64, globy::Int64, anzx::Int64, anzy::Int64, sprung::Int64)
        matrix=zeros(anzx,anzy)
        l=1
        for i in globx:sprung:globx+anzx*sprung-1
            d=1
            for j in globy:sprung:globy+anzy*sprung-1
                matrix[l,d]=f(i/(N+1),j/(N+1),k) 
                d=d+1
            end
            l=l+1
        end
        return matrix
    end

    #Hier werden die Raender eines Prozessors mit seinem rechten bzw linken NAchbarn getauscht
    function halo_oben_unten!(id::Int64, num_procs::Int64, x::Array{Float64,2}, nachbarn::Array{Int64,2})
        #Es soll nur was gemacht werden falls der Prozessor aktiv ist
        if nachbarn[2,2]!=-1
            anzx = size(x)[1]
            anzy = size(x)[2]
            #alle sachen die ausgetauscht werden muessen
            oben=x[2:2,2:anzy-1]
            unten=x[anzx-1:anzx-1, 2:anzy-1]
            obenbek=zeros(oben)
            untenbek=zeros(unten)
            #Tag einer nachricht ist SenderID*5000+EmpfaengerID
            #Beim schicken wird darauf geachtet, ob der Prozessor ein Rand ist. Wenn dieses der Fall ist soll er nur empfangen oder schicken.
            if nachbarn[1,2]!=-1 || nachbarn[3,2]!=-1
                #nach oben schicken
                if nachbarn[1,2]==-1
                    MPI.Recv!(untenbek,nachbarn[3,2],nachbarn[3,2]*5000+id,MPI.COMM_WORLD)
                    x[anzx:anzx,2:anzy-1]=untenbek
                elseif nachbarn[3,2]==-1
                    send=MPI.Isend(oben,nachbarn[1,2],id*5000+nachbarn[1,2],MPI.COMM_WORLD)
                    MPI.Waitall!([send])
                else
                    send=MPI.Isend(oben,nachbarn[1,2],id*5000+nachbarn[1,2],MPI.COMM_WORLD)
                    MPI.Recv!(untenbek,nachbarn[3,2],nachbarn[3,2]*5000+id,MPI.COMM_WORLD)
                    MPI.Waitall!([send])
                    x[anzx:anzx,2:anzy-1]=untenbek
                end
                #nach unten schicken
                if nachbarn[3,2]==-1
                    MPI.Recv!(obenbek,nachbarn[1,2],nachbarn[1,2]*5000+id,MPI.COMM_WORLD)
                    x[1:1,2:anzy-1]=obenbek
                elseif nachbarn[1,2]==-1
                    send=MPI.Isend(unten,nachbarn[3,2],id*5000+nachbarn[3,2],MPI.COMM_WORLD)
                    MPI.Waitall!([send])
                else
                    send=MPI.Isend(unten,nachbarn[3,2],id*5000+nachbarn[3,2],MPI.COMM_WORLD)
                    MPI.Recv!(obenbek,nachbarn[1,2],nachbarn[1,2]*5000+id,MPI.COMM_WORLD)
                    MPI.Waitall!([send])
                    x[1:1,2:anzy-1]=obenbek
                end
            end
        end
    end

    #Hier werden die Raender eines Prozessors mit seinem rechten bzw linken NAchbarn getauscht
    function halo_rechts_links!(id::Int64, num_procs::Int64, x::Array{Float64,2}, nachbarn::Array{Int64,2})
        #Es soll nur was gemacht werden falls der Prozessor aktiv ist
        if nachbarn[2,2]!=-1
            anzx = size(x)[1]
            anzy = size(x)[2]
            #alle sachen die ausgetauscht werden muessen
            links=x[2:anzx-1, 2:2]
            rechts=x[2:anzx-1, anzy-1:anzy-1]
            linksbek=zeros(links)
            rechtsbek=zeros(rechts)
            #Tag einer nachricht ist SenderID*5000+EmpfaengerID
            #Beim schicken wird darauf geachtet, ob der Prozessor ein Rand ist. Wenn dieses der Fall ist soll er nur empfangen oder schicken.
            if nachbarn[2,3]!=-1 || nachbarn[2,1]!=-1
                #nach rechts schicken
                if nachbarn[2,3]==-1
                    MPI.Recv!(linksbek,nachbarn[2,1],nachbarn[2,1]*5000+id,MPI.COMM_WORLD)
                    x[2:anzx-1,1:1]=linksbek
                elseif nachbarn[2,1]==-1
                    send=MPI.Isend(rechts,nachbarn[2,3],id*5000+nachbarn[2,3],MPI.COMM_WORLD)
                    MPI.Waitall!([send])
                else
                    send=MPI.Isend(rechts,nachbarn[2,3],id*5000+nachbarn[2,3],MPI.COMM_WORLD)
                    MPI.Recv!(linksbek,nachbarn[2,1],nachbarn[2,1]*5000+id,MPI.COMM_WORLD)
                    MPI.Waitall!([send])
                    x[2:anzx-1,1:1]=linksbek
                end
                #nach links schicken
                if nachbarn[2,1]==-1
                    MPI.Recv!(rechtsbek,nachbarn[2,3],nachbarn[2,3]*5000+id,MPI.COMM_WORLD)
                    x[2:anzx-1,anzy:anzy]=rechtsbek
                elseif nachbarn[2,3]==-1
                    send=MPI.Isend(links,nachbarn[2,1],id*5000+nachbarn[2,1],MPI.COMM_WORLD)
                    MPI.Waitall!([send])
                else
                    send=MPI.Isend(links,nachbarn[2,1],id*5000+nachbarn[2,1],MPI.COMM_WORLD)
                    MPI.Recv!(rechtsbek,nachbarn[2,3],nachbarn[2,3]*5000+id,MPI.COMM_WORLD)
                    MPI.Waitall!([send])
                    x[2:anzx-1,anzy:anzy]=rechtsbek
                end
            end
        end
    end
    
    function random_matrix(n::Int64, koords::Array{Int64,1}, groesse::Array{Int64,1})
        srand(33)
        return rand(n,n)[koords[1]:koords[1]+groesse[1]-1,koords[2]:koords[2]+groesse[2]-1]
    end
     

end

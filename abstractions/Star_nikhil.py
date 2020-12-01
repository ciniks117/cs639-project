import numpy as np
import scipy.linalg

class Star:
        # init method or constructor   
        def __init__(self,V,C,d):  
                    [nV, mV] = V.shape
                    [nC, mC] = C.shape
                    [nd, md] = d.shape

                    if mV != mC+1:
                        error('Inconsistency between basic matrix and constraint matrix')

                    if nC != nd:
                        error('Inconsistency between constraint matrix and constraint vector')

                    if md != 1:
                        error('constraint vector should have one column');

                    self.V = V;
                    self.C = C; 
                    self.d = d;
                                                            

        """
        # affine mapping of star set S = Wx + b;
        def affineMap(self, W, b):
            # @W: mapping matrix
            # @b: mapping vector
            # @S: new star set
            
            if size(W, 2) ~= obj.dim
                error('Inconsistency between the affine mapping matrix and dimension of the star set');
            end
            
            if ~isempty(b)
                if size(b, 1) ~= size(W, 1)
                    error('Inconsistency between the mapping vec and mapping matrix');
                end

                if size(b, 2) ~= 1
                    error('Mapping vector should have one column');
                end

                newV = W * obj.V;
                newV(:, 1) = newV(:, 1) + b;
            else
                newV = W * obj.V;
            end
            
            if ~isempty(obj.predicate_lb)
                S = Star(newV, obj.C, obj.d, obj.predicate_lb, obj.predicate_ub);
            else
                S = Star(newV, obj.C, obj.d);
            end
        """

        # scalar map of a Star S' = alp * S, 0 <= alp <= alp_max
        def scalarMap(self, alp_max):
            """
            % @a_max: maximum value of a
            % @S: new Star
            
            % note: we always require that alp >= 0
            % =============================================================
            % S: x = alp*c + V* alph * a, Ca <= d
            % note that:   Ca <= d -> C*alph*a <= alp*a <= alp_max * d
            % let: beta = alp * a, we have
            % S := x = alp * c + V * beta, C * beta <= alp_max * d,
            %                              0 <= alp <= alp_max
            % Let g = [beta; alp]
            % S = Star(new_V, new_C, new_d), where:
            %   new_V = [0 c V], new_C = [0 -1; 0 1; 0 C], new_d = [0; alpha_max; alp_max * d]
            %       
            % S has one more basic vector compared with obj
            % =============================================================
            """
            dim=2
            new_c = np.zeros[(dim,1)]
            new_V = np.c_[self.V, new_c]
            new_C = scipy.linalg.block_diag(obj.C, [-1, 1])
            new_d = np.vstack((alp_max*self.d, 0, alp_max))
            S = Star(new_V, new_C, new_d)
            return S

        
        # concatenate many stars
        def concatenateStars(stars):
            # @stars: an array of stars
            
            new_c = []
            new_V = []
            new_C = []
            new_d = []
            
            n = length(stars);
            
            for i in range(1,n):
                #if ~isa(stars(i), 'Star'):
                #    error('The %d th input is not a Star', i)
        
                new_c = np.vstack((new_c, stars[i].V[:,1]))
                new_V = scipy.linalg.block_diag(new_V, stars(i).V[:, 2:stars(i).nVar + 1])
                new_C = scipy.linalg.block_diag(new_C, stars(i).C);
                new_d = np.vstack(new_d, stars[i].d)
                
            
            S = Star(np.c_[new_c, new_V], new_C, new_d)
            return S
           
       
        """ 
        # merge stars using boxes and overlapness
        def merge_stars(I, nS, parallel):
            
            # @I: array of stars
            # @nP: number of stars of the output S
            
            n = length(I)
            B = []
            if strcmp(parallel, 'single'):
                
                for i=1:n
                    B = [B I(i).getBox];
                end

                m = I(1).dim;

                n = length(B);

                C = zeros(n, 2*m);
                for i=1:n
                    C(i, :) = [B(i).lb' B(i).ub'];
                end

                idx = kmeans(C, nS); % clustering boxes into nP groups

                R = cell(nS, 1);

                for i=1:nS
                    for j=1:n
                        if idx(j) == i
                            R{i, 1} = [R{i, 1} B(j)];
                        end
                    end
                end

                S = [];
                for i=1:nS
                    B = Box.boxHull(R{i, 1}); % return a box                    
                    S = [S B.toStar];
                end
                
            elif strcmp(parallel, 'parallel'):
                
                for i=1:n
                    B = [B I(i).getBox];

                m = I(1).dim;

                n = length(B);

                C = zeros(n, 2*m);
                for i=1:n
                    C(i, :) = [B(i).lb' B(i).ub'];

                idx = kmeans(C, nS); % clustering boxes into nP groups

                R = cell(nS, 1);

                for i=1:nS
                    for j=1:n
                        if idx(j) == i
                            R{i, 1} = [R{i, 1} B(j)];

                S = [];
                for i=1:nS
                    B = Box.boxHull(R{i, 1}); % return a box                    
                    S = [S B.toStar];
                end
                
            else:
                error('Unknown parallel option');

        """


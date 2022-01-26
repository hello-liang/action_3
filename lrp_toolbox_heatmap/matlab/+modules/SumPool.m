classdef SumPool < modules.Module
    % @author: Sebastian Lapuschkin
    % @author: Gregoire Montavon
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
    % @date: 09.11.2016
    % @version: 1.2+
    % @copyright: Copyright (c) 2016-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
    % @license : BSD-2-Clause
    %
    % Rectification Layer

    properties
        %layer parameters
        stride
        pool

        %temporary variables
        Y
        X
    end

    methods
        function obj = SumPool(pool,stride)
            % Constructor
            %
            % Parameters
            % ----------
            %
            % pool : (h,w)
            %     the size of the pooling mask in vertical (h) and horizontal (w) direction
            %
            % stride : (h,w)
            %     the vertical (h) and horizontal (w) step sizes between filter applications.

            obj = obj@modules.Module();

            if nargin < 2 || (exist('stride','var') && isempty(stride))
                obj.stride = [2,2];
            else
                obj.stride = stride;
            end
            if nargin < 1 || (exist('pool','var') && isempty(pool))
                obj.pool = [2,2];
            else
                obj.pool = pool;
            end
        end


        function Y = forward(obj,X)
            % Realizes the forward pass of an input through the sum pooling layer.
            %
            % Parameters
            % ----------
            % X : matrix
            %     a network input, shaped (N,H,W,D), with
            %     N = batch size
            %     H, W, D = input size in heigth, width, depth
            %
            % Returns
            % -------
            % Y : matrix
            %     the sum-pooled outputs, reduced in size due to given stride and pooling size

            obj.X = X;
            [N,H,W,D]= size(X);

            hpool = obj.pool(1);        wpool = obj.pool(2);
            hstride = obj.stride(1);    wstride = obj.stride(2);

            %assume the given pooling and stride parameters are carefully
            %chosen
            Hout = (H - hpool)/hstride + 1;
            Wout = (W - wpool)/wstride + 1;

            normalizer = 1./sqrt(hpool*wpool);

            %initialize output
            obj.Y = zeros(N,Hout,Wout,D);
            for i = 1:Hout
               for j = 1:Wout
                  obj.Y(:,i,j,:) = sum(sum(X(:,(i-1)*hstride+1:(i-1)*hstride+hpool,(j-1)*wstride+1:(j-1)*wstride+wpool,:),2),3) .* normalizer; %normalizer to produce well-conditioned output values
               end
            end
            Y = obj.Y; %'return'
        end



        function DX = backward(obj,DY)
            % Backward-passes an input error gradient DY towards the input neurons of this sum pooling layer.
            %
            % Parameters
            % ----------
            %
            % DY : matrix
            %     an error gradient shaped same as the output array of forward, i.e. (N,Hy,Wy,Dy) with
            %     N = number of samples in the batch
            %     Hy = heigth of the output
            %     Wy = width of the output
            %     Dy = output depth = input depth
            %
            %
            % Returns
            % -------
            %
            % DX : matrix
            %     the error gradient propagated towards the input

            [N,H,W,D] = size(obj.X);

            hpool = obj.pool(1);        wpool = obj.pool(2);
            hstride = obj.stride(1);    wstride = obj.stride(2);

            %assume the given pooling and stride parameters are carefully
            %chosen
            Hout = (H - hpool)/hstride + 1;
            Wout = (W - wpool)/wstride + 1;


            normalizer = 1./sqrt(hpool * wpool);

            %distribute the gradient (1 * DY) towards all contributing
            %inputs evenly
            DX = zeros(N,H,W,D);
            for i = 1:Hout
                for j = 1:Wout
                    dx = DX(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);
                    dy = repmat(DY(:,i,j,:),[1 hpool wpool 1]);
                    DX(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :) = (dx + dy) .* normalizer;
                end
            end
        end


        function clean(obj)
           obj.X = [];
           obj.Y = [];
        end



        function Rx = simple_lrp(obj,R)
            % LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
            [N,H,W,D] = size(obj.X);

            hpool = obj.pool(1);        wpool = obj.pool(2);
            hstride = obj.stride(1);    wstride = obj.stride(2);

            %assume the given pooling and stride parameters are carefully
            %chosen
            Hout = (H - hpool)/hstride + 1;
            Wout = (W - wpool)/wstride + 1;

            Rx = zeros(N,H,W,D);
            for i = 1:Hout
                for j = 1:Wout
                    Z = obj.X(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);
                    Zs = sum(sum(Z,2),3);
                    Zs = Zs + 1e-12 .* ((Zs >= 0) .* 2 -1); % add a weak numerical stabilizer to avoid zero division

                    rr = repmat(R(:,i,j,:),[1,hpool,wpool,1]);
                    zz = Z ./ repmat(Zs,[1,hpool,wpool,1]);
                    rx = Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);

                    Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :) = rx + rr .* zz;
                end
            end
        end

        function Rx = flat_lrp(obj,R)
            % distribute relevance for each output evenly to the output neurons' receptive fields.
            [N,H,W,D] = size(obj.X);

            hpool = obj.pool(1);        wpool = obj.pool(2);
            hstride = obj.stride(1);    wstride = obj.stride(2);

            %assume the given pooling and stride parameters are carefully
            %chosen
            Hout = (H - hpool)/hstride + 1;
            Wout = (W - wpool)/wstride + 1;

            Rx = zeros(N,H,W,D);
            for i = 1:Hout
                for j = 1:Wout
                    Z = ones(N,hpool,wpool,D);
                    Zs = sum(sum(Z,2),3);

                    rr = repmat(R(:,i,j,:),[1,hpool,wpool,1]);
                    zz = Z ./ repmat(Zs,[1,hpool,wpool,1]);
                    rx = Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);

                    Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :) = rx + rr .* zz;
                end
            end
        end

        function Rx = ww_lrp(obj,R)
            % due to uniform weights used for sum pooling (1), this method defaults to flat_lrp(R)
            Rx = obj.flat_lrp(R);
        end


        function Rx = epsilon_lrp(obj,R,epsilon)
            % LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
            [N,H,W,D] = size(obj.X);

            hpool = obj.pool(1);        wpool = obj.pool(2);
            hstride = obj.stride(1);    wstride = obj.stride(2);

            %assume the given pooling and stride parameters are carefully
            %chosen
            Hout = (H - hpool)/hstride + 1;
            Wout = (W - wpool)/wstride + 1;

            Rx = zeros(N,H,W,D);
            for i = 1:Hout
                for j = 1:Wout
                    Z = obj.X(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);
                    Zs = sum(sum(Z,2),3);
                    Zs = Zs + epsilon .* ((Zs >= 0) .* 2 -1); % add epsilon stabilizer to avoid zero division

                    rr = repmat(R(:,i,j,:),[1,hpool,wpool,1]);
                    zz = Z ./ repmat(Zs,[1,hpool,wpool,1]);
                    rx = Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);

                    Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :) = rx + rr .* zz;
                end
            end
        end

        function Rx = alphabeta_lrp(obj,R,alpha)
            % LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
            [N,H,W,D] = size(obj.X);

            hpool = obj.pool(1);        wpool = obj.pool(2);
            hstride = obj.stride(1);    wstride = obj.stride(2);

            %assume the given pooling and stride parameters are carefully
            %chosen
            Hout = (H - hpool)/hstride + 1;
            Wout = (W - wpool)/wstride + 1;


            beta = 1 - alpha;
            Rx = zeros(N,H,W,D);
            for i = 1:Hout
                for j = 1:Wout
                    Z = obj.X(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);
                    rr = repmat(R(:,i,j,:),[1,hpool,wpool,1]);

                    if ~(alpha == 0)
                        Zp = Z .* (Z > 0);
                        Zsp = sum(sum(Zp,2),3) + 1e-16; %zero division is quite likely in sum pooling layers when using the alpha-variant
                        Ralpha = alpha .* rr .* (Zp ./ repmat(Zsp,[1,hpool,wpool,1]));
                    else
                        Ralpha = 0;
                    end


                    if ~(beta == 0)
                        Zn = Z .* (Z < 0);
                        Zsn = sum(sum(Zn,2),3) - 1e-16; %zero division is quite likely in sum pooling layers when using the alpha-variant
                        Rbeta = beta .* rr .* (Zn ./ repmat(Zsn,[1,hpool,wpool,1]));
                    else
                        Rbeta = 0;
                    end

                    rx = Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);
                    Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :) = rx + Ralpha + Rbeta;
                end
            end
        end
    end
end
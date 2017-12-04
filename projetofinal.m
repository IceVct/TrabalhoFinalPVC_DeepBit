%% Projeto final de PVC - 2/2017 
%Andre Luis Souto Ferreira - 140016261
%Victor Araujo Vieira - 14/0032801

%% Script que executa os comandos necessarios para aplicar o modelo deepbit treinado nas imagens de insetos
% Sera dividido em 4 partes: Inicializacao dos dados, extracao dos
% descritores binarios para todas imagens,
% treinamento dos SVM e, por ultimo, o teste da eficiencia dos classificadores.

%% Inicializacao dos dados
% Parte do script que vai preparar os dados

close all;
clear;

addpath(genpath(pwd));

% variaveis do caffe
addpath('/home/victor/UnB/Semestre8/PVC/Projetos/ProjetoFinal/cvpr16-deepbit/matlab');
% Mude firsttime para 1, caso seja a primeira vez que esteja rodando o
% codigo
firstTime = 0;

if(firstTime == 1)
    % modelo deepbit
    model_file = '/home/victor/UnB/Semestre8/PVC/Projetos/ProjetoFinal/cvpr16-deepbit/models/deepbit/DeepBit32_final_iter_1.caffemodel';
    % definicao do modelo
    model_def_file = '/home/victor/UnB/Semestre8/PVC/Projetos/ProjetoFinal/cvpr16-deepbit/models/deepbit/deploy32.prototxt';

    caffe.set_mode_gpu();
    caffe.set_device(0);
    net = caffe.Net(model_def_file, model_file, 'test');
    net.blobs('data').reshape([224 224 3 1]); % reshape blob 'data'

    mediaBin = 0; % media que vai ser usada para a binarizacao dos atributos extraidos pelo deepbit

    todasImagens = '/home/victor/UnB/Semestre8/PVC/Projetos/ProjetoFinal/Imagens/todasimagens.txt';
    listaImagens = read_cell(todasImagens);

    %% Extracao dos descritores binarios para todas as imagens e calculo da media mediaBin

    numImagens = length(listaImagens);

    % Loop que vai ler as imagens da lista de imagens e fazer as operacoes
    % necessarias

    resultDeepBit = zeros(numImagens, 32);

    % Se for a primeira vez que esta rodando, faz todo o procedimento, se nao for,
    % ja carrega o objeto mat
    % Para cada imagem, vai rodar o modelo deepbit e vai adicinar o resultado
    % em um vetor coluna, de modo que vai ficar 4900x32
    for i = 1:numImagens
        im = imread(listaImagens{i});
        im = imresize(im, [224, 224]);
        resultModel = net.forward({im});
        resultDeepBit(i, :) = resultModel{1, 1};
    end
    feat_result_file = sprintf('%s/resultDeepBit.mat', '.');
    save(feat_result_file);

else
    load('./resultDeepBit.mat');
end

% Calcula a media de cada coluna
mediaCol = mean(resultDeepBit);
% Calcula o resultado da media de cada coluna, ou seja, calcula agora a
% media geral
mediaBin = mean(mediaCol);


% Avalia os features originais de cada img, se for maior que a media geral
% vira 1, senao 0
binarioImagens = (resultDeepBit > mediaBin);
binarioImagens = double(binarioImagens); % converte de logical para double

%% Treinamento da SVM sem cross validation

% Le o arquivo que contem as labels de todas as classes
classes = read_cell('./classes.txt');

% Usando o resultado ao avaliar todas as imagens no modelo DeepBit
% Ao inves de usar os descritores binarios 
SVMClassifier = fitcecoc(resultDeepBit, classes);

%% Treinamento da SVM com cross validation 

rng(1)
t = templateSVM('Standardize', 1);
SVMClassifierCrossVal = fitcecoc(resultDeepBit,classes,'Learners',t,...
'ClassNames',{'Coleoptera','Diptera','Hemiptera', 'Hymenoptera', 'Isoptera', 'Lepidoptera', 'Orthoptera'});
CVClassifier = crossval(SVMClassifierCrossVal);


%% Teste da eficiencia dos classificadores

% testando classificador sem cross validation
[label, score] = resubPredict(SVMClassifier);

% testando classificador com cross validation
resultCV = kfoldPredict(CVClassifier);

%% Treinamento da SVM sem cross validation usando descritores binarios em si 
SVMBin1 = fitcecoc(binarioImagens, classes);
[labelBin1, scoreBin1] = resubPredict(SVMBin1);

%% Treinamento da SVM com cross validation usando descritores binarios em si
rng(1)
tbin = templateSVM('Standardize', 1);
SVMClassifierCrossValBin = fitcecoc(binarioImagens,classes,'Learners',tbin,...
'ClassNames',{'Coleoptera','Diptera','Hemiptera', 'Hymenoptera', 'Isoptera', 'Lepidoptera', 'Orthoptera'});
CVClassifierBin = crossval(SVMClassifierCrossValBin);
resultCVBin = kfoldPredict(CVClassifierBin);

%% Mostrando as matrizes de confusao

% Passos para a criacao da matriz de confusao
isLabels = unique(classes);
nLabels = numel(isLabels);
[n,p] = size(resultDeepBit);

[~,grpLabel] = ismember(label,isLabels); 
labelMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpLabel,(1:n)'); 
labelMat(idxLinear) = 1; % Flags the row corresponding to the class 
[~,grpY] = ismember(classes,isLabels); 
YMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
YMat(idxLinearY) = 1;

[~,grpLabel2] = ismember(resultCV,isLabels); 
labelMat2 = zeros(nLabels,n);
idxLinear2 = sub2ind([nLabels n],grpLabel2,(1:n)'); 
labelMat2(idxLinear2) = 1; % Flags the row corresponding to the class 
[~,grpY2] = ismember(classes,isLabels); 
YMat2 = zeros(nLabels,n); 
idxLinearY2 = sub2ind([nLabels n],grpY2,(1:n)'); 
YMat2(idxLinearY2) = 1; 

[~,grpLabel3] = ismember(labelBin1,isLabels); 
labelMat3 = zeros(nLabels,n); 
idxLinear3 = sub2ind([nLabels n],grpLabel3,(1:n)'); 
labelMat3(idxLinear3) = 1; % Flags the row corresponding to the class 
[~,grpY3] = ismember(classes,isLabels); 
YMat3 = zeros(nLabels,n); 
idxLinearY3 = sub2ind([nLabels n],grpY3,(1:n)'); 
YMat3(idxLinearY3) = 1;

[~,grpLabel4] = ismember(resultCVBin,isLabels); 
labelMat4 = zeros(nLabels,n);
idxLinear4 = sub2ind([nLabels n],grpLabel4,(1:n)'); 
labelMat4(idxLinear4) = 1; % Flags the row corresponding to the class 
[~,grpY4] = ismember(classes,isLabels); 
YMat4 = zeros(nLabels,n); 
idxLinearY4 = sub2ind([nLabels n],grpY4,(1:n)'); 
YMat4(idxLinearY4) = 1; 

% Imprime as matrizes de confusao
figure;
plotconfusion(YMat,labelMat);

figure;
plotconfusion(YMat2,labelMat2);

figure;
plotconfusion(YMat3,labelMat3);

figure;
plotconfusion(YMat4,labelMat4);

function make_3col_files(subject)

maindir = pwd;
datadir = fullfile(maindir,'data',num2str(subject));
outputdir = fullfile(maindir,'evfiles_m02',num2str(subject));
if ~exist(outputdir,'dir')
    mkdir(outputdir);
end

addpath(fullfile(maindir,'rw'));


%{

> data(1)

ans =
          Npoints: 7 % how many points they would've earned
           lapse1: 0 % did the fail to respond in phase 1?
           lapse2: 0 % did the fail to respond in phase 2?
       deckchoice: 2 % what did they choose?
              RT1: 0.5860
              RT2: 0.5221
     choice_onset: 4.0241 % when they were presented the options
     press1_onset: 4.6101 % when did they press it
       info_onset: 8.0149
    partner_onset: 12.2563
     press2_onset: 12.7785
        aff_onset: 16.5313
          soc_win: 0
             word: 'active' % unsure what this is
           rating: 0
         is_catch: 0
         
%}

blocks = 1:4;
for r = 1:length(blocks)
    
    load(fullfile(datadir,sprintf('%s_feedback_%d.mat',num2str(subject),r)))
    choicedata = [data.Npoints; data.deckchoice]';
    
    % estimate learning
    out = choicedata(:,1);
    out(out==4) = 2;
    out = out-1;
    dec = choicedata(:,2);
    
    alpha = 0.30;
    beta = 2;
    [~, ~, ~, pe_e] = runRW(dec, out, alpha, beta);
    
    ntrials = length(data);
    
    %make empty mats (for *_par, will make *_con last)
    selfchoice = zeros(ntrials,3);
    inf_par = zeros(ntrials,3);
    aff_pos_par = zeros(ntrials,3);
    aff_neg_par = zeros(ntrials,3);
    partnerchoice = zeros(ntrials,3);
    lapse1 = zeros(ntrials,3);
    lapse2 = zeros(ntrials,3);
    
    for t = 1:ntrials
        if data(t).is_catch
            continue
        end
        if data(t).lapse1
            lapse1(t,1) = data(t).choice_onset;
            lapse1(t,2) = 2.5;
            lapse1(t,3) = 1;
        else
            selfchoice(t,1) = data(t).choice_onset;
            selfchoice(t,2) = data(t).RT1;
            selfchoice(t,3) = 1;
            
            inf_par(t,1) = data(t).info_onset;
            inf_par(t,2) = 1.5;
            inf_par(t,3) = abs(pe_e(t));
            
            if data(t).lapse2
                lapse2(t,1) = data(t).partner_onset;
                lapse2(t,2) = 2.5;
                lapse2(t,3) = 1;
            else
                partnerchoice(t,1) = data(t).partner_onset;
                partnerchoice(t,2) = data(t).RT2;
                partnerchoice(t,3) = 1;
                
                if data(t).soc_win
                    aff_pos_par(t,1) = data(t).aff_onset;
                    aff_pos_par(t,2) = 1.5;
                    aff_pos_par(t,3) = data(t).Npoints;
                else
                    aff_neg_par(t,1) = data(t).aff_onset;
                    aff_neg_par(t,2) = 1.5;
                    aff_neg_par(t,3) = data(t).Npoints;
                end
            end
        end
    end
    
    partnerchoice(~partnerchoice(:,1),:) = [];
    selfchoice(~selfchoice(:,1),:) = [];
    lapse2(~lapse2(:,1),:) = [];
    lapse1(~lapse1(:,1),:) = [];
    
    % demean parametric regressors
    inf_par(~inf_par(:,1),:) = [];
    inf_par(:,3) = inf_par(:,3)-mean(inf_par(:,3));
    aff_neg_par(~aff_neg_par(:,1),:) = [];
    aff_neg_par(:,3) = aff_neg_par(:,3)-mean(aff_neg_par(:,3));
    aff_pos_par(~aff_pos_par(:,1),:) = [];
    aff_pos_par(:,3) = aff_pos_par(:,3)-mean(aff_pos_par(:,3));
    
    % make constants
    inf_con = inf_par;
    inf_con(:,3) = 1;
    aff_neg_con = aff_neg_par;
    aff_neg_con(:,3) = 1;
    aff_pos_con = aff_pos_par;
    aff_pos_con(:,3) = 1;
    
    cd(outputdir);
    dlmwrite(sprintf('inf_con%d.txt',r),inf_con,'delimiter','\t','precision','%.6f')
    dlmwrite(sprintf('aff_neg_con%d.txt',r),aff_neg_con,'delimiter','\t','precision','%.6f')
    dlmwrite(sprintf('aff_pos_con%d.txt',r),aff_pos_con,'delimiter','\t','precision','%.6f')
    dlmwrite(sprintf('inf_par%d.txt',r),inf_par,'delimiter','\t','precision','%.6f')
    dlmwrite(sprintf('aff_neg_par%d.txt',r),aff_neg_par,'delimiter','\t','precision','%.6f')
    dlmwrite(sprintf('aff_pos_par%d.txt',r),aff_pos_par,'delimiter','\t','precision','%.6f')
    dlmwrite(sprintf('partnerchoice%d.txt',r),partnerchoice,'delimiter','\t','precision','%.6f')
    dlmwrite(sprintf('selfchoice%d.txt',r),selfchoice,'delimiter','\t','precision','%.6f')
    cd(maindir);
end


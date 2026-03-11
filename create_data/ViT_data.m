% ======================================================================
%  MATLAB + QuaDRiGa | 3GPP TR 38.901 UMa NLOS CSI Generator (NO NOISE)
%  - Scenario: 3GPP_38.901_UMa_NLOS
%  - Clusters: 21 | Rays per cluster: 20
%  - UE init position randomized | linear track
%  - Speeds uniformly distributed in [10, 100] km/h (Num_Scenes scenes)
%  - BS: 8x8 UPA, dual-pol (ports colocated) -> 128 Tx ports
%  - UE: 1x4 ULA -> 4 Rx ports
%  - Per TTI: H_t (K_RB × Nr × Nt) saved as .mat
% ======================================================================

clear; clc;

%% 0) Global Settings
SEED = 20251102; rng(SEED,'twister');

BASE_DIR = 'D:/Program Files/fangzhen/ViT_data_last_mimo/';   % 根目录（每个速度一个文件夹）
SAVE_REAL_IMAG_SPLIT = false;    % true: 另存 H_real/H_imag
DO_PLOTS  = false;               % 可选可视化

SCENARIO_ID = '3GPP_38.901_UMa_NLOS';

% -------- System / OFDM --------
fc    = 2.4e9;            % 2.4 GHz
K_RB  = 64;               % 64 RB（你也可以改成 96）
RB_bw = 180e3;            % 180 kHz
BW_total = K_RB * RB_bw;  % total BW

% -------- Time --------
Snaps_per_Scene = 1000;   % 每个场景快照数（你也可以改成 2000）
dt     = 0.5e-3;          % 0.5 ms
T_total= Snaps_per_Scene * dt;

% -------- Scenes / Speeds --------
Num_Scenes = 20;
speeds_kmh = linspace(10, 100, Num_Scenes);   % 10~100均匀

% -------- Geometry --------
BS_height_m = 25.0;
UE_height_m = 1.5;

% -------- Antenna config --------
BS_Nh = 8; BS_Nv = 8;     % ★ 改这里：8x8 UPA
BS_dual_pol = true;       % true -> 8*8*2=128 ports
BS_d_lambda = 0.5;        % 0.5 λ

UE_ports    = 4;          % UE 4 根天线（1x4 ULA）
UE_d_lambda = 0.5;        % 0.5 λ

% -------- 38.901 small-scale params --------
Ncl = 21;                 % number of clusters
Nray = 20;                % rays per cluster

if ~exist(BASE_DIR,'dir'), mkdir(BASE_DIR); end

%% 1) Derived
c0 = 299792458; lambda = c0/fc;

fprintf('=== CONFIG SUMMARY ===\n');
fprintf('Scenario : %s\n', SCENARIO_ID);
fprintf('fc       : %.3f GHz | BW: %.2f MHz (%d RB @ 180 kHz)\n', fc/1e9, BW_total/1e6, K_RB);
fprintf('dt       : %.3f ms | snaps/scene: %d | scenes: %d\n', dt*1e3, Snaps_per_Scene, Num_Scenes);
fprintf('Speed    : uniform in [10,100] km/h\n');
fprintf('Clusters : %d | Rays/cluster: %d\n', Ncl, Nray);
fprintf('UE ports : %d | BS dual-pol: %d\n\n', UE_ports, BS_dual_pol);

%% 2) QuaDRiGa Simulation Parameters
sp = qd_simulation_parameters;
sp.center_frequency = fc;

%% 3) Build BS / UE arrays (robust build)
% ---- BS array ----
d_bs = BS_d_lambda*lambda;
if BS_dual_pol
    BS_ports = 2*BS_Nh*BS_Nv;   % = 128
else
    BS_ports = BS_Nh*BS_Nv;     % = 64
end

bs = qd_arrayant('omni');
for n = 2:BS_ports, bs.copy_element(1,n); end
pos_bs = zeros(3,BS_ports); idx=0;

for iv=0:BS_Nv-1
  for ih=0:BS_Nh-1
    idx=idx+1; pos_bs(:,idx) = [ih*d_bs; iv*d_bs; 0];
    if BS_dual_pol
        idx=idx+1; pos_bs(:,idx) = [ih*d_bs; iv*d_bs; 0];
    end
  end
end
bs.element_position = pos_bs;

% ---- UE array (1×4 ULA) ----
d_ue = UE_d_lambda*lambda;
ue = qd_arrayant('omni');
for n=2:UE_ports, ue.copy_element(1,n); end
pos_ue = zeros(3,UE_ports);
for i=0:UE_ports-1
    pos_ue(:,i+1) = [i*d_ue; 0; 0];
end
ue.element_position = pos_ue;

%% 4) Main loop over scenes (different speeds)
for i_sc = 1:Num_Scenes

    speed_kmh = speeds_kmh(i_sc);
    v_ms = speed_kmh/3.6;

    folder_name = sprintf('%dkmh_%d', round(speed_kmh), Snaps_per_Scene);
    OUT_DIR = fullfile(BASE_DIR, folder_name);
    if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

    fprintf('--- [Scene %02d/%02d] speed=%.1f km/h | saving to %s ---\n', ...
        i_sc, Num_Scenes, speed_kmh, folder_name);

    %% 4.1) Layout
    l  = qd_layout; 
    l.simpar = sp;
    l.no_tx = 1; 
    l.no_rx = 1;

    l.tx_position(:,1) = [0; 0; BS_height_m];
    l.rx_position(:,1) = [0; 0; UE_height_m];

    l.tx_array = bs;
    l.rx_array = ue;

    % UE initial position randomized (radius 50~500m, random angle)
    phi = rand*2*pi; 
    r = 50 + rand*450;
    l.rx_position(:,1) = [r*cos(phi); r*sin(phi); UE_height_m];

    %% 4.2) Mobility track (linear)
    path_len = v_ms*T_total;
    rx_track = qd_track('linear', path_len);
    rx_track.set_speed(v_ms);
    rx_track.initial_position = l.rx_position(:,1);
    rx_track.interpolate_positions(dt);
    l.rx_track = rx_track;

    % Set scenario AFTER track is attached
    l.set_scenario(SCENARIO_ID);

    %% 4.3) Force 21 clusters & 20 rays/cluster via builder
    builder = l.init_builder;

    % 先生成默认参数（确保结构体齐全）
    builder.gen_parameters;

    % 尝试写入簇/射线配置（不同版本字段名可能不同）
    if isfield(builder,'scenpar')
        if isfield(builder.scenpar,'NumClusters')
            builder.scenpar.NumClusters = Ncl;
        end
        if isfield(builder.scenpar,'NumSubPaths')
            builder.scenpar.NumSubPaths = Nray;
        end
        if isfield(builder.scenpar,'NumRaysPerCluster')
            builder.scenpar.NumRaysPerCluster = Nray;
        end
    end

    % 重新生成参数以使修改生效
    builder.gen_parameters;

    % Get channels
    ch = builder.get_channels;

    % Align snapshots
    if ch(1).no_snap ~= Snaps_per_Scene
        t_new = (0:Snaps_per_Scene-1).' * dt;
        ch = ch.interpolate(t_new);
    end

    %% 4.4) Frequency response & reshape to (T, K, Nr, Nt)
    Hf = ch.fr(BW_total, K_RB);

    dims = size(Hf); 
    fprintf('   Hf shape: [%s]\n', num2str(dims));

    if numel(dims)==4 && dims(3)==K_RB && dims(4)==Snaps_per_Scene
        H = permute(Hf,[4,3,1,2]);     % (Nr,Nt,K,T) -> (T,K,Nr,Nt)
    elseif numel(dims)==4 && dims(4)==K_RB && dims(3)==Snaps_per_Scene
        H = permute(Hf,[3,4,1,2]);     % (Nr,Nt,T,K) -> (T,K,Nr,Nt)
    else
        error('Unexpected Hf shape, check size(Hf).');
    end

    assert(isequal(size(H), [Snaps_per_Scene, K_RB, UE_ports, BS_ports]), 'H size mismatch');

    %% 4.5) Save per-TTI files (NO NOISE)
    subc_spacing = BW_total / K_RB;
    freq_offsets_Hz = ((0:K_RB-1) - (K_RB-1)/2) * subc_spacing;

    for t = 1:Snaps_per_Scene
        H_t = reshape(H(t,:,:,:), [K_RB, UE_ports, BS_ports]);  % (K, Nr, Nt)

        meta = struct();
        meta.tti_index        = t-1;
        meta.center_frequency = fc;
        meta.K_RB             = K_RB;
        meta.RB_spacing_Hz    = RB_bw;
        meta.dt_s             = dt;
        meta.speed_kmh        = speed_kmh;
        meta.v_ms             = v_ms;
        meta.UE_ports         = UE_ports;
        meta.BS_ports         = BS_ports;
        meta.H_shape          = sprintf('(K, Nr, Nt) = (%d, %d, %d)', K_RB, UE_ports, BS_ports);
        meta.scenario         = SCENARIO_ID;
        meta.NumClusters      = Ncl;
        meta.RaysPerCluster   = Nray;
        meta.subc_spacing_Hz  = subc_spacing;
        meta.freq_offsets_Hz  = freq_offsets_Hz;

        fname = fullfile(OUT_DIR, sprintf('%d_TTi.mat', t-1));
        if SAVE_REAL_IMAG_SPLIT
            H_real = real(H_t); H_imag = imag(H_t);
            save(fname,'H_real','H_imag','meta','-v7.3');
        else
            save(fname,'H_t','meta','-v7.3');
        end
    end

    fprintf('   ✅ Scene done: %s | H_t size = [%d %d %d]\n', folder_name, K_RB, UE_ports, BS_ports);
end

fprintf('\n✅ Finished: %d scenes × %d TTIs saved to\n%s\n', Num_Scenes, Snaps_per_Scene, BASE_DIR);
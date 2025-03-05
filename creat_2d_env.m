% 创建电磁仿真环境

clc;
clear;

viewer = siteviewer(Buildings="chicago.osm");

pm = propagationModel( ...
    "raytracing", ...
    "Method","sbr", ...
    "CoordinateSystem","geographic",...
    "MaxNumReflections",3, ...
    "MaxNumDiffractions",1 ...
    );

% lat + 上
% lat - 下
% lon + 右
% lon - 左
tx = txsite( ...
    "Name","source", ...
    "CoordinateSystem","geographic",...
    Latitude=41.8810  , ...
    Longitude=-87.6305 ...
    );
% show(tx)

outputFolder = "d:\vscode_py\reinforcement_learning\outputData"; % 可根据需要修改
if ~isfolder(outputFolder)
    mkdir(outputFolder)
end

for latVal = (tx.Latitude - 0.004):0.0001:(tx.Latitude + 0.004)
    for lonVal = (tx.Longitude - 0.004):0.0001:(tx.Longitude + 0.004)
        tx.Latitude = latVal;
        tx.Longitude = lonVal;
        % show(tx)

        % 定义网格参数
        gridSize = 50; % 每个轴的网格点数
        latRange = linspace(tx.Latitude - 0.001, tx.Latitude + 0.001, gridSize); % 纬度范围
        lonRange = linspace(tx.Longitude - 0.001, tx.Longitude + 0.001, gridSize); % 经度范围
        [latGrid, lonGrid] = ndgrid(latRange, lonRange);

        T = readgeotable("chicago.osm",layer="buildings");
        % sea2ground = 245-51;

        % 初始化接收功率矩阵，碰撞标识矩阵
        powerGrid = zeros(size(latGrid));
        insideBuildingGrid = zeros(size(latGrid)); % 建筑物标识矩阵 (1: 内部, 0: 外部)

        % 获取tx的高度
        txAlt = 0;
        tx_location = geopointshape(tx.Latitude, tx.Longitude);
        for i = 1:height(T)
            if isinterior(T.Shape(i), tx_location)
                txAlt = T.MaxHeight(i) + tx.AntennaHeight;
                break;
            end
        end
        
        % 遍历每个网格点计算功率
        for ix = 1:gridSize
            for iy = 1:gridSize
                % 定义接收机位置
                rxLat = latGrid(ix, iy);
                rxLon = lonGrid(ix, iy);
                rxAlt = txAlt;
                
                % 初始化功率变量
                power = -Inf;

                % 检查点是否位于建筑物上
                location = geopointshape(latGrid(ix, iy), lonGrid(ix, iy));
                haveBuilding = false; % 默认没有建筑物
                isInside = false; % 默认不在建筑物内部

                for i = 1:height(T)
                    if isinterior(T.Shape(i), location)
                        haveBuilding = true;
                        buildingHeight = T.MaxHeight(i);
                        if (rxAlt >= buildingHeight)
                            rxAntennaHeight = rxAlt - buildingHeight;
                            % 在障碍物上方
                            rx = rxsite( ...
                                "Name", "receiver", ...
                                "CoordinateSystem", "geographic", ...
                                "Latitude", rxLat, ...
                                "Longitude", rxLon, ...
                                "AntennaHeight", rxAntennaHeight);
                            show(rx);
                            % 计算接收功率dBm（分贝毫瓦）P_dBm=10⋅log_10(P_W/1mW),P_W是接收到的功率（以瓦特为单位）
                            power = sigstrength(rx,tx,pm);
                        else
                            isInside = true;
                        end
                        break;
                    end
                end
                
                % 没有建筑物
                if ~haveBuilding
                    rx = rxsite("CoordinateSystem", "geographic", ...
                                    "Latitude", rxLat, ...
                                    "Longitude", rxLon, ...
                                    "AntennaHeight", rxAlt);
                    show(rx);
                    power = sigstrength(rx,tx,pm);
                end

                % 存储功率值和碰撞标识
                powerGrid(ix, iy) = power;
                insideBuildingGrid(ix, iy) = isInside;

            end
        end

        powerGrid(powerGrid == -Inf) = NaN;

        % 计算功率网格的梯度
        [gradLat, gradLon] = gradient(powerGrid);

        gradLat = fillmissing(gradLat, 'nearest');
        gradLon = fillmissing(gradLon, 'nearest');

        powerGrid = fillmissing(powerGrid, 'nearest');

        % 计算障碍物外部的距离场
        outsideDistance = bwdist(insideBuildingGrid);
        % 计算障碍物内部的距离场
        insideDistance = bwdist(~insideBuildingGrid);
        % 将障碍物内部的距离取负值
        ESDF = outsideDistance;
        ESDF(insideBuildingGrid == 1) = -insideDistance(insideBuildingGrid == 1);

        % 计算 ESDF 矩阵的梯度
        [gradESDFLat, gradESDFLon] = gradient(ESDF);

        % 以经纬度命名保存文件
        filename = sprintf('gridData_lat%.4f_lon%.4f.mat', latVal, lonVal);
        filepathToSave = fullfile(outputFolder, filename);
        save(filepathToSave, 'powerGrid', 'insideBuildingGrid', ...
            'gradLat', 'gradLon', 'ESDF', ...
            'gradESDFLat', 'gradESDFLon');
    end
end

close(viewer);

package com.jinghui.api.service;

import com.alibaba.fastjson.JSONObject;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.jinghui.api.dto.ModelUtilization5CountDto;
import com.jinghui.api.entity.InterpretationResult;
import org.springframework.web.multipart.MultipartFile;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public interface IInterpretationResultService extends IService<InterpretationResult> {

    /**
     * 获取数据解译文件记录(分页查询)
     * @param page
     * @return
     */
    Page<InterpretationResult> getPageList(Page<InterpretationResult> page);

    /**
     * 删除数据解译文件记录
     * @param id
     * @return
     */
    int deleteInterpretationResult(int id);

    /**
     * 分析结果并保存
     * @param dataId
     * @param modelId
     * @return
     */
    int analyseData(int dataId, int modelId);

    /**
     * 获取绘制信息
     * @param id
     * @return
     */
    HashMap<String, String> getVisMessage(int id);

    /**
     * 分析数据并可视化
     * @param modelId
     * @param visOptions
     * @return
     */
    JSONObject analyseAndVisualizeData(int modelId, MultipartFile file, List<Integer> visOptions) throws IOException;

    /**
     * 用于测试与python算法库的接口联调
     * @param params
     * @return
     */
    JSONObject sayHelloToPython(String params);

    /**
     * 通过路径读取图片
     * @param fileName
     * @return
     */
    byte[] getFileURLPictures(String fileName) throws IOException;

    int getAllCount();

    /**
     * 获取当日新增解译总数
     * @return
     */
    int getCountInCurrentDay();

    /**
     * 获取模型利用比例
     */
    List<ModelUtilization5CountDto> getModelUtilizationRatio();

    /**
     * 获取近7天的模型比例
     * @return
     */
    JSONObject getShip7Count();
}

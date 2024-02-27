package com.jinghui.api.service.impl;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.jinghui.api.dto.ModelUtilization5CountDto;
import com.jinghui.api.entity.Data;
import com.jinghui.api.entity.InterpretationResult;
import com.jinghui.api.entity.Model;
import com.jinghui.api.mapper.DataMapper;
import com.jinghui.api.mapper.InterpretationResultMapper;
import com.jinghui.api.mapper.ModelMapper;
import com.jinghui.api.service.IDataService;
import com.jinghui.api.service.IInterpretationResultService;
import com.jinghui.api.util.DateTimeUtil;
import com.jinghui.api.util.JsonUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * @Description: TODO
 * @author: scott
 * @date: 2022年01月09日 10:35
 */
@Service
public class InterpretationResultImpl extends ServiceImpl<InterpretationResultMapper, InterpretationResult> implements IInterpretationResultService {
    @Value("${file.json-dir}")
    private String jsonDir;

    @Value("${file.cache-dir}")
    private String cacheDir;

    @Autowired
    InterpretationResultMapper interpretationResultMapper;

    @Autowired
    DataMapper dataMapper;

    @Autowired
    ModelMapper modelMapper;

    @Autowired
    IDataService dataService;

    @Autowired
    private RestTemplate restTemplate;

    @Override
    public Page<InterpretationResult> getPageList(Page<InterpretationResult> page) {
        return interpretationResultMapper.selectPage(page, null);
    }

    @Override
    public int deleteInterpretationResult(int id) {
        return interpretationResultMapper.deleteById(id);
    }

    @Override
    public int analyseData(int dataId, int modelId) {
        InterpretationResult interpretationResult = new InterpretationResult();
        interpretationResult.setDataId(dataId);
        interpretationResult.setModelId(modelId);

        Date date = new Date();
        interpretationResult.setCreateTime(date);

        //TODO： 转到算法模型库进行数据分析并返回json结果文件路径
        //生成uuid
        try {
            String uuid = UUID.randomUUID().toString().replaceAll("-", "");
            interpretationResult.setJsonUrl(jsonDir+"\\"+uuid+".json");
        } catch (Exception e){
            e.printStackTrace();
            return -1;
        }
        interpretationResultMapper.insert(interpretationResult);

        return interpretationResult.getId();
    }

    @Override
    public HashMap<String, String> getVisMessage(int id) {
        // 查询匹配的分析结果记录
        InterpretationResult result = interpretationResultMapper.selectById(id);
        // 查询匹配的数据文件记录
        Data data = dataMapper.selectById(result.getDataId());
        // 查询匹配的模型文件记录
        Model model = modelMapper.selectById(result.getModelId());

        // 时间转换
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        HashMap<String, String> jsonResult = new HashMap<>();
        jsonResult.put("dataCreateTime", formatter.format(data.getCreateTime()));
        jsonResult.put("resultCreateTime", formatter.format(result.getCreateTime()));
        jsonResult.put("modelName", model.getName());
        jsonResult.put("dataImageUrl", data.getFileUrl());

        return jsonResult;
    }

    @Override
    @Transactional
    public JSONObject analyseAndVisualizeData(int modelId, MultipartFile file, List<Integer> visOptions) throws IOException {
        // 先保存图片
        Data data = new Data();
        data.setDescription("由快捷操作快速添加的数据");
        int dataId = dataService.addSingleData(data, file);

        // 保存解译记录
        // 请求算法解译
        //请求地址
        String url = "http://localhost:8000/analyseAndVisualizeData/";
        // 读取所需数据
        Model model = modelMapper.selectById(modelId);
        String[] arr = model.getFileUrl().split("\\\\");
        String modelFilename = arr[arr.length-1];
        //入参
        JSONObject postData = new JSONObject();
        postData.put("model_filename", modelFilename);
        postData.put("group_id", dataId);
        postData.put("vis_options", visOptions);
        // 响应消息
        JSONObject json = restTemplate.postForObject(url, postData, JSONObject.class);
        // 保存解译记录
        InterpretationResult interpretationResult = new InterpretationResult();
        interpretationResult.setModelId(modelId);
        interpretationResult.setDataId(dataId);
        Date date = new Date();
        interpretationResult.setCreateTime(date);
        List<String> jsonUrlList = (List<String>)json.get("resultJsonPaths");
        interpretationResult.setJsonUrl(jsonUrlList.get(0));
        interpretationResultMapper.insert(interpretationResult);

        return json;
    }

    @Override
    public JSONObject sayHelloToPython(String params) {
        //请求地址
        String url = "http://localhost:8000/sayHello/";
        //入参
        JSONObject postData = new JSONObject();
        postData.put("params", params);

        JSONObject json = restTemplate.postForObject(url, postData, JSONObject.class);

        return json;
    }

    @Override
    public byte[] getFileURLPictures(String fileName) throws IOException {
        String url = cacheDir + "/" + fileName;
        File file = new File(url);
        FileInputStream picInput = new FileInputStream(file);
        byte[] buffer = new byte[picInput.available()];
        picInput.read(buffer, 0, picInput.available());
        picInput.close();
        return buffer;
    }

    @Override
    public int getAllCount() {
        return interpretationResultMapper.selectCount(null);
    }

    @Override
    public int getCountInCurrentDay() {
        QueryWrapper<InterpretationResult> queryWrapper = new QueryWrapper<>();
        Date currentTime = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");
        String rqq = formatter.format(currentTime);
        queryWrapper.apply("DATE(create_time) >= STR_TO_DATE('"+rqq+"00:00:00','%Y-%m-%d %H:%i:%s')");
        return interpretationResultMapper.selectCount(queryWrapper);
    }

    @Override
    public List<ModelUtilization5CountDto> getModelUtilizationRatio() {
        return interpretationResultMapper.queryModelUtilization5Count();
    }

    @Override
    public JSONObject getShip7Count() {
        JSONObject result = new JSONObject();

        // 获取近7天日期
        List<String> days = DateTimeUtil.getIntervalsDate(7);

        // 获取近7天船舶数量
        List<Integer> counts = new ArrayList<>();
        for (int i=0; i<7; i++){
            // 读取JSON文件，统计船舶数量
            QueryWrapper<InterpretationResult> queryWrapper = new QueryWrapper<>();
            if (i > 0){
                queryWrapper.apply("DATE(create_time) <= STR_TO_DATE('"+days.get(i-1)+"23:59:59','%Y-%m-%d %H:%i:%s')");
            }
            queryWrapper.apply("DATE(create_time) >= STR_TO_DATE('"+days.get(i)+"00:00:00','%Y-%m-%d %H:%i:%s')");
            List<InterpretationResult> interpretationResults = interpretationResultMapper.selectList(queryWrapper);
            int count = 0;
            for (InterpretationResult item: interpretationResults){
                JSONObject jsonContent = JsonUtil.readJsonFile(item.getJsonUrl());
                JSONObject detResults = null;
                if (jsonContent != null) {
                    detResults = jsonContent.getJSONObject("det_results");
                    JSONArray mboxesScore = detResults.getJSONArray("mboxes_score");
                    count += mboxesScore.size();
                }
            }
            counts.add(count);
        }

        result.put("days", days);
        result.put("counts", counts);

        return result;
    }
}

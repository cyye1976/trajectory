package com.jinghui.api.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.jinghui.api.dto.ModelUtilization5CountDto;
import com.jinghui.api.entity.InterpretationResult;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface InterpretationResultMapper extends BaseMapper<InterpretationResult> {

    @Select("select MD.`name` as name, COUNT(*) AS value from interpretation_result as IR JOIN model as MD WHERE IR.model_id = MD.id GROUP BY MD.`name` LIMIT 5")
    List<ModelUtilization5CountDto> queryModelUtilization5Count();

    List<InterpretationResult> queryInterpretation();
}

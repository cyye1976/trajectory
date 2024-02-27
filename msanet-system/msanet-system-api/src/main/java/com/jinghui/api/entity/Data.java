package com.jinghui.api.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;


/**
 * @Description: TODO
 * @author: scott
 * @date: 2022年01月07日 9:54
 */
@lombok.Data
@ApiModel
public class Data {
    @TableId(type = IdType.AUTO)
    @ApiModelProperty("ID")
    private int Id;
    @ApiModelProperty("数据批量分组ID")
    private int groupId;
    @ApiModelProperty("数据文件路径")
    private String fileUrl;
    @ApiModelProperty("描述")
    private String description;
    @ApiModelProperty("创建时间")
    @JsonFormat(timezone = "GMT+8",pattern = "yyyy-MM-dd HH:mm:ss")
    @DateTimeFormat(pattern="yyyy-MM-dd HH:mm:ss")
    private Date createTime;
}

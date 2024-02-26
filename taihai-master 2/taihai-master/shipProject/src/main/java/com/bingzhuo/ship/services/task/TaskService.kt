package com.bingzhuo.ship.services.task

import com.bingzhuo.ship.entity.dto.TaskDto
import com.bingzhuo.ship.mapper.TaskMapper
import com.bingzhuo.ship.tools.FileTool
import com.bingzhuo.ship.entity.po.Task
import com.bingzhuo.ship.services.task.annotation.CheckRunning
import com.bingzhuo.ship.services.task.command.impl.*
import org.springframework.core.env.Environment
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import org.springframework.web.multipart.MultipartFile
import java.io.File
import java.lang.RuntimeException
import java.util.*
import javax.annotation.Resource

@Service
open class TaskService {
    @Resource
    open lateinit var taskMapper: TaskMapper
    @Resource
    open lateinit var env: Environment


    @Transactional(propagation = Propagation.REQUIRED)
    open fun creatTask(taskDto:TaskDto):Task{
        val task = Task()
        taskMapper.saveTask(task)
        val outPath = env.getProperty("web.task-path")+"/"+task.taskId+"/"+"dataset/"
        FileTool.unZip(FileTool.saveFile(taskDto.uplaodFile!!,
                env.getProperty("web.upload-path") + UUID.randomUUID().toString().substring(0..8)+"_${task.taskId}.zip"),
                outPath)
        return taskMapper.getTaskById(task.taskId!!);
    }

    @CheckRunning
    @Transactional
    open fun traceSplit(taskId:Int):String{
        val task = taskMapper.getTaskById(taskId)?:throw RuntimeException("无该任务")
        val inputPath = env.getProperty("web.task-path")+taskId+"/dataset/"
        val outputPath = env.getProperty("web.task-path")+taskId+"/trace_split"+"/out/"
        val command = SplitPythonCommand(inputPath, outputPath, env.getProperty("web.python-file-path"), env.getProperty("web.python-exe-path"))
        val result = command.execute()
        task.isSplited = true
        taskMapper.updateTask(task)
        return result.replace(env.getProperty("web.task-path"),"")
    }

    @CheckRunning
    @Transactional
    open fun featureExtract(taskId:Int):List<String>{
        val task = taskMapper.getTaskById(taskId)?:throw RuntimeException("无该任务")
        val inputPath1 = env.getProperty("web.task-path")+taskId+"/trace_split"+"/out/"
        val outputPath1 =  env.getProperty("web.task-path")+taskId+"/trace_time_convert"+"/out/"

        if (!File(inputPath1).exists())
            throw RuntimeException("请先执行前置处理")

        val inputPath2 = outputPath1 + ',' + inputPath1
        val outputPath2 = env.getProperty("web.task-path")+taskId+"/trace_feature_project"+"/out/"

        val inputPath3 = outputPath2
        val outputPath3 = env.getProperty("web.task-path")+taskId+"/to_one"+"/out/"

        val inputPath4 = outputPath3
        val outputPath4 = env.getProperty("web.task-path")+taskId+"/feature_classify"+"/out/"

        val inputPath5 = outputPath4
        val outputPath5 = env.getProperty("web.task-path")+taskId+"/self_encode"+"/out/"

        val convertCommand = TraceTimeCovertPythonCommand(inputPath1, outputPath1,
                env.getProperty("web.python-file-path"),
                env.getProperty("web.python-exe-path"))
        convertCommand.execute()

        val featureProjectCommand = TraceFeatureProjectCommand(inputPath2, outputPath2,
                env.getProperty("web.python-file-path"),
                env.getProperty("web.python-exe-path"))
        featureProjectCommand.execute()

        val toOneCommand = ToOneCommand(inputPath3, outputPath3,
                env.getProperty("web.python-file-path"),
                env.getProperty("web.python-exe-path"))
        toOneCommand.execute()

        val traceClassifyCommand = TraceFeatureClassify(inputPath4, outputPath4,
                env.getProperty("web.python-file-path"),
                env.getProperty("web.python-exe-path"))
        traceClassifyCommand.execute()

        val selfEncode = SelfEncode(inputPath5, outputPath5,
                env.getProperty("web.python-file-path"),
                env.getProperty("web.python-exe-path"))
        val result = selfEncode.execute()

        task.isFeatureExacted = true
        taskMapper.updateTask(task)
        return result.map { it.replace(env.getProperty("web.task-path"),"") }
    }

    @CheckRunning
    @Transactional
    open fun gmmCluster(taskId:Int):Map<String, String>{
        val task = taskMapper.getTaskById(taskId)?:throw RuntimeException("无该任务")
        val inputPath = env.getProperty("web.task-path")+taskId+"/self_encode"+"/out/"+','+env.getProperty("web.task-path")+taskId+"/feature_classify"+"/out/"
        val outputPath = env.getProperty("web.task-path")+taskId+"/GMM"+"/out/"
        if (!File(env.getProperty("web.task-path")+taskId+"/self_encode"+"/out/").exists())
            throw RuntimeException("请先执行前置处理")
        val gmmCommand = GMMCluster(inputPath, outputPath,
                env.getProperty("web.python-file-path"),
                env.getProperty("web.python-exe-path"))
        val result =  gmmCommand.execute()
        task.isClustered = true
        taskMapper.updateTask(task)
        return result
    }
}
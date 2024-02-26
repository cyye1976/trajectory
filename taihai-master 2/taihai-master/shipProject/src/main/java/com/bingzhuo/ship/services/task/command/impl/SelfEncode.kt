package com.bingzhuo.ship.services.task.command.impl

import com.bingzhuo.ship.services.task.command.inter.AbstractPythonCommand
import java.util.concurrent.TimeUnit

class SelfEncode(val inputPath:String?, val outputPath:String?, var pyFilePrefix: String = "",var allPrefix: String = ""): AbstractPythonCommand<List<String>>() {
    override fun getPrefix(): String {
        return allPrefix
    }

    override fun getAdditionalParams(): Array<String> {
        if(inputPath!=null && outputPath!=null)
            return arrayOf("--inputPath",inputPath, "--outputPath", outputPath)
        else
            return arrayOf("")
    }

    override fun getCommand(): String {
        return "${getPyFilePathPrifix()}5.py"
    }

    override fun getTimeOut(): Long {
        return 10
    }

    override fun getTimeUnit(): TimeUnit {
        return TimeUnit.MINUTES
    }

    override fun dealResult(cmdOut: String): List<String> {
        val resultList = cmdOut.split("\n").filter { e->e.startsWith("<outputFigPath>:") }
        return resultList
    }

    override fun getPyFilePathPrifix(): String {
        return pyFilePrefix
    }
}
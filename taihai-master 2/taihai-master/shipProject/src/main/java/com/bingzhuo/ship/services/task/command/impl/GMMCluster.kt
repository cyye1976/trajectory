package com.bingzhuo.ship.services.task.command.impl

import com.bingzhuo.ship.services.task.command.inter.AbstractPythonCommand
import java.util.concurrent.TimeUnit

class GMMCluster(val inputPath:String?, val outputPath:String?, var pyFilePrefix: String = "",var allPrefix: String = ""): AbstractPythonCommand<Map<String,String>>() {
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
        return "${getPyFilePathPrifix()}6.py"
    }

    override fun getTimeOut(): Long {
        return 10
    }

    override fun getTimeUnit(): TimeUnit {
        return TimeUnit.MINUTES
    }

    override fun dealResult(cmdOut: String): Map<String,String> {
        val regex = Regex("""A.+A""")
        var resultMap:MutableMap<String,String> = HashMap<String,String>()
        cmdOut.split("\n").filter { e->regex.containsMatchIn(e) }
                .map {  val key = regex.find(it)?.value;
                        val value = it.replace(key.toString(),"")
                        listOf(key,value) }
                    .forEach{resultMap.put(it[0] as String,it[1] as String)}
        return resultMap
    }

    override fun getPyFilePathPrifix(): String {
        return pyFilePrefix
    }
}
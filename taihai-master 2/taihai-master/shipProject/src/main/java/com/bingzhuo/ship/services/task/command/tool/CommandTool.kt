package com.bingzhuo.ship.services.task.command.tool

import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.*
import java.util.concurrent.TimeUnit
import kotlin.collections.ArrayList


object CommandTool {
    fun execCommand(command:String, timeOut:Long, timeUnit:TimeUnit):String{
        val st = StringTokenizer(command)
        val cmdarray:MutableList<String> = ArrayList()
        var i = 0
        while(st.hasMoreTokens()){
            cmdarray.add(st.nextToken())
        }
        val processBuilder = ProcessBuilder(cmdarray.toList())
        processBuilder.redirectErrorStream(true)
        val proc = processBuilder.start()
        //val proc = Runtime.getRuntime().exec(command)
        //用输入输出流来截取结果
        val `in` = BufferedReader(InputStreamReader(proc.inputStream))
        val stringBuilder = StringBuilder()
        var line: String? = null
        while (`in`.readLine().also { line = it } != null) {
            stringBuilder.append(line+"\n")
        }
        `in`.close()
        val status = proc.waitFor(timeOut, timeUnit)
        if(status==false)
            throw RuntimeException("命令执行超时")
        if(proc.exitValue() != 0)
            throw RuntimeException(stringBuilder.toString())
        return stringBuilder.toString()
    }
}
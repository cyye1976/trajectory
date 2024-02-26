package com.bingzhuo.ship.entity.po


data class Task(var taskId:Int? = null, var filePath:String? = null, var isSplited:Boolean=false,
                var isFeatureExacted:Boolean=false,
                var isClustered:Boolean=false) {

}
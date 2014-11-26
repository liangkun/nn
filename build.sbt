// Copyright (c) 2014 Liang Kun. All Rights Reserved.
// Authors: Liang Kun <liangkun@data-intelli.com>

import AssemblyKeys._

name := "sann"

description := "Simple Artificial Neural Network"

organization := "org.sann"

version := "0.1"

scalaVersion := "2.11.4"

scalacOptions ++= Seq("-deprecation", "-unchecked", "-feature", "-optimise")

libraryDependencies ++= {
  val breezeVersion = "0.10"
  Seq(
    "org.scalanlp" %% "breeze" % breezeVersion,
    "org.scalanlp" %% "breeze-natives" % breezeVersion,
    "org.scalanlp" %% "breeze-viz" % "0.9",
    "org.scalatest" %% "scalatest" % "2.2.1" % "test",
    "org.testng" % "testng" % "6.1.1" % "test"
  )
}

mainClass in Global := Some("org.sann.Main")

jarName in assembly := s"sann-${version.value}.jar"

assemblySettings
